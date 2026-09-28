"""Дообучение multilingual-e5-small на парах из train и кодирование текстов.

Модель intfloat/multilingual-e5-small (лицензия MIT), веса лежат в models/.
Учим на парах "запрос + фильтры -> объявление (заголовок, параметры, начало описания)".
Функция потерь InfoNCE, негативы - остальные объявления в батче.
Контрольные запросы из split.py в обучение не попадают.

После обучения кодируем корпус benchmark_items, все тексты запросов train
(нужны для поиска похожих запросов) и запросы контроля и бенчмарка.
"""
import argparse
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from solution import ROOT, SEARCH, norm, log

CACHE = ROOT / "cache"
BASE = ROOT / "models" / "multilingual-e5-small"
TUNED = ROOT / "models" / "e5-small-avito"
SEED = 20260928
# Без видеокарты тоже работает, только медленнее
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def doc_text(title, params, desc):
    # Сначала заголовок, потом параметры, описание обрезаем
    return f"passage: {title}. {params}. {str(desc)[:300]}"


def query_text(q, filters=""):
    q = norm(q)
    return f"query: {q}. {filters}" if filters else f"query: {q}"


def mean_pool(hidden, mask):
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return F.normalize((hidden * mask).sum(1) / mask.sum(1).clamp(min=1), dim=-1)


def train(epochs=1, batch=128, lr=5e-5, temperature=0.05, max_steps=2000):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    cols = SEARCH + ["item_id", "item_title_raw", "item_description_raw", "item_infm_params_text"]
    df = pd.read_parquet(ROOT / "data/train.parquet", columns=cols)
    df = df[np.load(CACHE / "learn_mask.npy")]
    df["q"] = [query_text(q, f) for q, f in zip(df.search_query, df.search_infm_params_text)]
    df = df.drop_duplicates(["q", "item_id"]).reset_index(drop=True)
    docs = df.drop_duplicates("item_id").set_index("item_id")
    docs = pd.Series([doc_text(*r) for r in docs[["item_title_raw", "item_infm_params_text",
                                                  "item_description_raw"]].itertuples(index=False)],
                     index=docs.index)
    item_code = pd.factorize(df.item_id)[0]
    text_code = pd.factorize(df.search_query.map(norm))[0]
    log("pairs", len(df), "items", len(docs))

    device = DEVICE
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModel.from_pretrained(BASE).to(device)
    # Эмбеддинги словаря - это 96 млн из 118 млн параметров. Их замораживаем,
    # иначе состояние Adam не влезает в 6 ГБ видеопамяти
    model.embeddings.word_embeddings.weight.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    # 2000 шагов - это около 45 минут на GTX 1660 SUPER
    steps = min(max_steps, epochs * (len(df) // batch))
    # Прогрев 300 шагов, дальше lr линейно падает до нуля
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(s / 300, max(0.0, (steps - s) / (steps - 300))))
    model.train()
    step, t0 = 0, time.time()
    for ep in range(epochs):
        order = rng.permutation(len(df))
        for b in range(min(len(df) // batch, steps - step)):
            idx = order[b * batch:(b + 1) * batch]
            q = tok(df.q.values[idx].tolist(), padding=True, truncation=True, max_length=40,
                    return_tensors="pt").to(device)
            d = tok(docs.loc[df.item_id.values[idx]].tolist(), padding=True, truncation=True,
                    max_length=64, return_tensors="pt").to(device)
            # Считаем в fp32: на этой видеокарте fp16 почему-то в 5 раз медленнее
            qv = mean_pool(model(**q).last_hidden_state, q["attention_mask"])
            dv = mean_pool(model(**d).last_hidden_state, d["attention_mask"])
            logits = (qv @ dv.T) / temperature
            # Если в батче то же объявление или тот же текст запроса, это не негатив
            ic = torch.as_tensor(item_code[idx], device=device)
            tc = torch.as_tensor(text_code[idx], device=device)
            same = (ic[:, None] == ic[None, :]) | (tc[:, None] == tc[None, :])
            eye = torch.eye(len(idx), dtype=torch.bool, device=device)
            logits = logits.masked_fill(same & ~eye, -1e4)
            labels = torch.arange(len(idx), device=device)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                el = time.time() - t0
                log(f"step {step}/{steps} loss {loss.item():.3f} eta {el / step * (steps - step) / 60:.1f} min")
    TUNED.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(TUNED)
    tok.save_pretrained(TUNED)
    log("saved", TUNED)


@torch.inference_mode()
def encode(model, tok, texts, max_length, batch=256):
    texts = list(texts)
    out = np.empty((len(texts), model.config.hidden_size), np.float16)
    # Сортируем по длине, чтобы в батче было меньше паддинга
    order = np.argsort([len(t) for t in texts])
    for s in range(0, len(texts), batch):
        idx = order[s:s + batch]
        x = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_length,
                return_tensors="pt").to(DEVICE)
        v = mean_pool(model(**x).last_hidden_state, x["attention_mask"])
        out[idx] = v.cpu().numpy()
        if (s // batch) % 100 == 0:
            log("encoded", s, "/", len(texts))
    return out


def encode_all(path=TUNED, tag="tuned"):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path).to(DEVICE).eval()
    items = pd.read_parquet(ROOT / "data/benchmark_items.parquet",
                            columns=["item_id", "item_title_raw", "item_infm_params_text", "item_description_raw"])
    log("encode items", len(items))
    np.save(CACHE / f"{tag}_items.npy", encode(model, tok, [doc_text(*r) for r in items[
        ["item_title_raw", "item_infm_params_text", "item_description_raw"]].itertuples(index=False)], 64))
    texts = sorted(set(pd.read_parquet(ROOT / "data/train.parquet", columns=["search_query"]).search_query.map(norm)))
    pd.DataFrame({"text": texts}).to_parquet(CACHE / f"{tag}_train_texts.parquet", index=False)
    log("encode train texts", len(texts))
    np.save(CACHE / f"{tag}_train_texts.npy", encode(model, tok, [query_text(t) for t in texts], 40))
    for name, frame in [("control", pd.read_parquet(CACHE / "control.parquet")),
                        ("benchmark", pd.read_parquet(ROOT / "data/benchmark_queries.parquet"))]:
        # С фильтрами - для поиска объявлений, без фильтров - для поиска похожих запросов
        np.save(CACHE / f"{tag}_{name}_queries.npy", encode(model, tok, [
            query_text(q, f) for q, f in zip(frame.search_query, frame.search_infm_params_text)], 40))
        np.save(CACHE / f"{tag}_{name}_plain.npy", encode(model, tok, [query_text(q) for q in frame.search_query], 40))
    log("encoded")


def recall_check(tag="tuned"):
    """Быстрая проверка: recall@K одного E5 на контроле, без географии и CatBoost."""
    items = pd.read_parquet(ROOT / "data/benchmark_items.parquet", columns=["item_id"])
    truth = joblib.load(CACHE / "truth.joblib")
    ctrl = pd.read_parquet(CACHE / "control.parquet")
    iv = torch.as_tensor(np.load(CACHE / f"{tag}_items.npy").astype(np.float32), device=DEVICE)
    qv = torch.as_tensor(np.load(CACHE / f"{tag}_control_queries.npy").astype(np.float32), device=DEVICE)
    pos = {v: k for k, v in enumerate(items.item_id)}
    res = {k: [] for k in [50, 200, 500, 1000]}
    for s in range(0, len(ctrl), 256):
        top = (qv[s:s + 256] @ iv.T).topk(1000, dim=1).indices.cpu().numpy()
        for j, row in enumerate(ctrl.iloc[s:s + 256].itertuples()):
            t = [pos[v] for v in truth[row.query_id]]
            for k in res:
                res[k].append(np.isin(t, top[j, :k]).mean())
    log(tag, {k: round(float(np.mean(v)), 4) for k, v in res.items()})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["train", "encode", "check"], default="train")
    p.add_argument("--epochs", type=int, default=1)
    a = p.parse_args()
    if a.stage == "train":
        train(a.epochs)
        encode_all()
        recall_check()
    elif a.stage == "encode":
        encode_all()
    else:
        recall_check()
