"""Основной скрипт: лексический поиск (solution.py) + дообученный E5 + CatBoost.

Стадии запускаются по очереди, каждая отдельным процессом, чтобы в памяти
не висели два индекса сразу:
  index    - строит лексические индексы для контроля и бенчмарка;
  control  - кандидаты и признаки для контрольных запросов из split.py;
  bench    - то же для запросов бенчмарка (индекс строится по всему train);
  fit      - учит CatBoost на контроле, пишет метрики и answer.csv.
"""
import argparse
import gc
import json

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits

from solution import (ROOT, SEARCH, FEATURES, Retriever, evaluate, log, make_features,
                      topk, validate_answer)

CACHE = ROOT / "cache"
TAG = "tuned"
EXTRA = ["dense", "dense_microcat", "dense_neighbor_item", "dense_query_similarity",
         "title_word_coverage", "description_word_coverage", "params_word_coverage", "title_word_cosine"]
ALL_FEATURES = FEATURES + EXTRA
NEEDED = SEARCH + ["item_id", "item_title_raw", "item_microcat_id", "item_location_id",
                   "item_latitude", "item_longitude"]


class DenseRetriever(Retriever):
    """Retriever, к которому добавлены оценки от E5."""

    def attach(self, name):
        # Векторы посчитаны заранее в dense.py
        self.item_vec = np.load(CACHE / f"{TAG}_items.npy").astype(np.float32)
        texts = pd.read_parquet(CACHE / f"{TAG}_train_texts.parquet").text
        pos = {t: k for k, t in enumerate(texts)}
        tv = np.load(CACHE / f"{TAG}_train_texts.npy")
        # Берем только тексты, по которым строился индекс. Для контроля там нет его же ответов
        self.text_vec = tv[[pos[t] for t in self.qtexts]].astype(np.float32)
        self.query_vec = np.load(CACHE / f"{TAG}_{name}_queries.npy").astype(np.float32)
        self.plain_vec = np.load(CACHE / f"{TAG}_{name}_plain.npy").astype(np.float32)

    def retrieve(self, row, pool=600, extra_scores=None):
        with threadpool_limits(limits=4, user_api="blas"):
            dense = self.item_vec @ self.query_vec[row.qi]
            sim = self.text_vec @ self.plain_vec[row.qi]
        # Похожие по смыслу запросы train: что по ним выбирали и из каких микрокатегорий
        near = topk(sim, 40)
        w = np.exp((sim[near] - sim[near[0]]) / 0.02)
        w /= w.sum()
        q = sparse.csr_matrix((w, (np.zeros(len(near), int), near)), shape=(1, len(self.qtexts)))
        extra = {
            "dense": dense,
            "dense_microcat": (q @ self.qmicro).toarray().ravel()[self.imicro],
            "dense_neighbor_item": (q @ self.qitem).toarray().ravel(),
            "dense_query_similarity": float(sim[near[0]]),
        }
        return super().retrieve(row, pool=pool, extra_scores=extra)


def build_index(name):
    """Индекс строится несколько минут, поэтому сохраняем его в кэш."""
    path = CACHE / f"{name}_index.joblib"
    if path.exists():
        return joblib.load(path)
    items = pd.read_parquet(ROOT / "data/benchmark_items.parquet")
    train = pd.read_parquet(ROOT / "data/train.parquet", columns=NEEDED)
    if name == "control":
        train = train[np.load(CACHE / "learn_mask.npy")]
    model = DenseRetriever(items, train)
    del items, train
    gc.collect()
    joblib.dump(model, path)
    return model


def features(name):
    if name == "control":
        queries = pd.read_parquet(CACHE / "control.parquet")
        truth = joblib.load(CACHE / "truth.joblib")
    else:
        queries = pd.read_parquet(ROOT / "data/benchmark_queries.parquet")
        truth = None
    # qi - номер строки запроса в массивах векторов E5
    queries = queries.assign(qi=np.arange(len(queries)))
    model = build_index(name)
    model.attach(name)
    x, y, g, c = make_features(model, queries, truth)
    joblib.dump((x, y, g, c, model.ids), CACHE / f"{name}_features.joblib")
    log("saved", name, x.shape)


def pool_report(c, queries, truth, ids):
    r = [len(set(ids[idx]) & set(truth[q])) / len(truth[q]) for q, idx in zip(queries.query_id, c)]
    log("pool recall", np.mean(r), "mean pool size", np.mean([len(v) for v in c]))


def fit(iterations=1500):
    from catboost import CatBoostClassifier

    control = pd.read_parquet(CACHE / "control.parquet")
    truth = joblib.load(CACHE / "truth.joblib")
    x, y, g, c, ids = joblib.load(CACHE / "control_features.joblib")
    pool_report(c, control, truth, ids)
    fit_mask = control.split.to_numpy()[g] == "fit"
    valq = control[control.split == "valid"].reset_index(drop=True)
    vc = [c[k] for k in np.flatnonzero(control.split.to_numpy() == "valid")]
    # Для сравнения: отбор по простой сумме оценок и по одному E5
    report = {"baseline_base_score": evaluate(x[~fit_mask, FEATURES.index("base_score")], vc, valq, truth, ids),
              "dense_only": evaluate(x[~fit_mask, ALL_FEATURES.index("dense")], vc, valq, truth, ids)}
    log(report)
    params = dict(depth=8, learning_rate=0.05, loss_function="Logloss", l2_leaf_reg=8,
                  thread_count=10, random_seed=20260928, verbose=100, allow_writing_files=False)
    model = CatBoostClassifier(iterations=iterations, **params)
    model.fit(x[fit_mask], y[fit_mask], eval_set=(x[~fit_mask], y[~fit_mask]), early_stopping_rounds=150)
    report["model"] = evaluate(model.predict_proba(x[~fit_mask])[:, 1], vc, valq, truth, ids)
    report["iterations"] = int(model.tree_count_)
    report["features"] = dict(sorted(zip(ALL_FEATURES, map(float, model.feature_importances_)),
                                     key=lambda t: -t[1]))
    log(json.dumps(report, ensure_ascii=False, indent=1))
    (ROOT / "validation_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Итоговую модель учим на всем контроле. Деревьев чуть больше, так как данных стало больше
    final = CatBoostClassifier(iterations=int(model.tree_count_ * 1.15), **params)
    final.fit(x, y)
    final.save_model(str(ROOT / "selector.cbm"))
    del x, y, g, c
    gc.collect()

    bq = pd.read_parquet(ROOT / "data/benchmark_queries.parquet")
    bx, _, bg, bc, bids = joblib.load(CACHE / "benchmark_features.joblib")
    s = final.predict_proba(bx)[:, 1]
    starts = np.r_[0, np.cumsum([len(v) for v in bc])]
    answer = [" ".join(bids[idx[topk(s[a:b], 50)]]) for idx, a, b in zip(bc, starts[:-1], starts[1:])]
    pd.DataFrame({"query_id": bq.query_id, "answer": answer}).to_csv(ROOT / "answer.csv", index=False,
                                                                     encoding="utf-8")
    validate_answer(ROOT / "answer.csv", bq, pd.read_parquet(ROOT / "data/benchmark_items.parquet",
                                                             columns=["item_id"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["index", "control", "bench", "fit"], required=True)
    a = p.parse_args()
    if a.stage == "index":
        build_index("control")
        gc.collect()
        build_index("benchmark")
    elif a.stage == "control":
        features("control")
    elif a.stage == "bench":
        features("benchmark")
    else:
        fit()
