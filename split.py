"""Локальная проверка, похожая на бенчмарк.

В первой версии нужные объявления контрольных запросов добавлялись в корпус
извне, и рядом с ними не было похожих объявлений из того же города. Поэтому
локально было 0.90, а на проверке 0.847. В настоящем корпусе на каждый запрос
приходится около 75 объявлений той же локации.

Теперь контрольные запросы берутся только из тех контекстов train, у которых
все выбранные объявления уже есть в benchmark_items, а ищем по самому
benchmark_items без добавлений. Доли подогнаны под бенчмарк: около 62% текстов
не встречаются в train, около 37% запросов с фильтрами.
"""
import joblib
import numpy as np
import pandas as pd

from solution import ROOT, SEARCH, norm, log

CACHE = ROOT / "cache"
N_CONTROL = 5000       # 4/5 на обучение CatBoost, 1/5 на оценку
UNSEEN_SHARE = 0.62    # доля текстов бенчмарка, которых нет в train
FILTER_SHARE = 0.37    # доля запросов бенчмарка с фильтрами
SEED = 20260928


def make_split():
    CACHE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    train = pd.read_parquet(ROOT / "data/train.parquet", columns=SEARCH + ["item_id"])
    items = pd.read_parquet(ROOT / "data/benchmark_items.parquet", columns=["item_id"])
    train["_text"] = train.search_query.map(norm)
    train["_in"] = train.item_id.isin(set(items.item_id))

    # Контекст = запрос со всеми параметрами поиска. Берем те, где все выбранные объявления есть в корпусе
    ctx = train.groupby(SEARCH, sort=False).agg(ok=("_in", "all"), _text=("_text", "first")).reset_index()
    ctx = ctx[ctx.ok & (ctx.search_is_delivery_search == 0)]
    # "Новый" текст - тот, что встречается в train только в этом контексте.
    # Так не приходится выкидывать из train тысячи строк популярных запросов
    n_ctx = train.drop_duplicates(SEARCH).groupby("_text").size()
    ctx["unseen"] = ctx._text.map(n_ctx).eq(1).to_numpy()
    # В бенчмарке тексты не повторяются, поэтому берем один контекст на текст
    ctx = ctx.iloc[rng.permutation(len(ctx))].drop_duplicates("_text")
    has_f = ctx.search_infm_params_text.ne("")
    parts = []
    for f, share_f in [(True, FILTER_SHARE), (False, 1 - FILTER_SHARE)]:
        for u, share_u in [(True, UNSEEN_SHARE), (False, 1 - UNSEEN_SHARE)]:
            pool = ctx[(has_f == f) & (ctx.unseen == u)]
            parts.append(pool.iloc[: int(round(N_CONTROL * share_f * share_u))])
    ctx = pd.concat(parts)
    ctx = ctx.iloc[rng.permutation(len(ctx))].reset_index(drop=True)
    ctx["split"] = np.where(np.arange(len(ctx)) % 5 == 0, "valid", "fit")
    ctx["query_id"] = [f"local_{j:010d}" for j in range(len(ctx))]

    # Контрольные контексты убираем из данных, на которых строится поиск
    keys = pd.MultiIndex.from_frame(ctx[SEARCH])
    held = pd.MultiIndex.from_frame(train[SEARCH]).isin(keys)
    learn_mask = ~held

    held_rows = train.loc[held].merge(ctx[SEARCH + ["query_id"]], on=SEARCH)
    truth = held_rows.groupby("query_id").item_id.agg(lambda x: sorted(set(x))).to_dict()
    # Пересчитываем флаг по тому, что реально осталось в train
    seen_texts = set(train.loc[learn_mask, "_text"])
    ctx["unseen"] = ~ctx._text.isin(seen_texts)

    ctx.drop(columns=["ok"]).to_parquet(CACHE / "control.parquet", index=False)
    joblib.dump(truth, CACHE / "truth.joblib")
    np.save(CACHE / "learn_mask.npy", np.asarray(learn_mask))
    log("control", len(ctx), "unseen", ctx.unseen.mean(), "filters", ctx.search_infm_params_text.ne("").mean(),
        "learn rows", int(learn_mask.sum()), "positives/query", np.mean([len(v) for v in truth.values()]))


if __name__ == "__main__":
    make_split()
