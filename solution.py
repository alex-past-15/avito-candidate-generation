"""Общая часть поиска: обработка текста, индексы по корпусу и признаки кандидатов.

Здесь лексический поиск (BM25 и символьные n-граммы), подсказки из похожих
запросов train и география. Смысловую часть на E5 добавляет pipeline.py.
"""

from functools import lru_cache
import os
from pathlib import Path
import re
import time

os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize as l2normalize
import Stemmer

ROOT = Path(__file__).resolve().parent
SEARCH = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]
STEMMER = Stemmer.Stemmer("russian")
STOP = set(
    "и в во на по с со из за для от до к у о об а или не это как при под над без".split()
)


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def norm(s):
    return " ".join(re.findall(r"[a-zа-я0-9]+", str(s).lower().replace("ё", "е")))


@lru_cache(maxsize=500000)
def stem(w):
    return STEMMER.stemWord(w)


def words(s):
    return " ".join(stem(w) for w in norm(s).split() if w not in STOP)


def topk(a, k):
    # Индексы k наибольших значений, по убыванию
    k = min(k, len(a))
    idx = np.argpartition(a, len(a) - k)[-k:]
    return idx[np.lexsort((idx, -a[idx]))]


def bm25(counts):
    """BM25, длина нормируется отдельно для каждого поля."""
    x = counts.tocsr().astype(np.float32)
    lengths = np.asarray(x.sum(axis=1)).ravel()
    df = np.bincount(x.indices, minlength=x.shape[1])
    idf = np.log1p((x.shape[0] - df + 0.5) / (df + 0.5)).astype(np.float32)
    denom = 1.2 * (0.65 + 0.35 * lengths / max(lengths.mean(), 1))
    x.data = x.data * 2.2 / (x.data + np.repeat(denom, np.diff(x.indptr)))
    return x.tocsc(), idf


class Retriever:
    def __init__(self, items, train):
        self.items = items.reset_index(drop=True)
        self.ids = self.items.item_id.to_numpy()
        self.idmap = {v: j for j, v in enumerate(self.ids)}
        self.n = len(items)
        log("Normalize corpus", self.n)
        self.titles = items.item_title_raw.map(norm).tolist()
        title = items.item_title_raw.map(words).tolist()
        desc = items.item_description_raw.str.slice(0, 12000).map(words).tolist()
        params = items.item_infm_params_text.str.slice(0, 4000).map(words).tolist()
        self.wv = CountVectorizer(
            tokenizer=str.split,
            token_pattern=None,
            lowercase=False,
            ngram_range=(1, 2),
            min_df=2,
            max_features=350000,
            dtype=np.float32,
        )
        log("Fit word vocabulary")
        self.wv.fit((a + " " + b + " " + c for a, b, c in zip(title, desc, params)))
        self.unigram = np.array([' ' not in t for t in self.wv.get_feature_names_out()])
        self.fields = []
        for name, texts in [
            ("title", title),
            ("description", desc),
            ("params", params),
        ]:
            log("Index", name)
            counts = self.wv.transform(texts)
            self.fields.append(bm25(counts))
            if name == "title":
                self.title_tfidf = l2normalize(
                    counts.multiply(self.fields[0][1])
                ).tocsr()
        del title, desc, params, counts
        log("Index character ngrams")
        self.cv = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=3,
            max_features=250000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.char = self.cv.fit_transform(self.titles).T.tocsr()

        log("Build behavioral features")
        train = train.copy()
        train["_text"] = train.search_query.map(norm)
        self.qtexts = sorted(train._text.unique())
        qmap = {v: j for j, v in enumerate(self.qtexts)}
        # Два способа найти похожие запросы train: по символам и по словам
        self.qcv = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=2,
            max_features=180000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.qchar = self.qcv.fit_transform(self.qtexts).T.tocsr()
        self.qwv = TfidfVectorizer(
            tokenizer=str.split,
            token_pattern=None,
            ngram_range=(1, 2),
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.qword = self.qwv.fit_transform([words(t) for t in self.qtexts]).T.tocsr()
        qr = train._text.map(qmap).to_numpy()

        # Какие микрокатегории выбирали по каждому тексту запроса
        self.micros = sorted(set(train.item_microcat_id) | set(items.item_microcat_id))
        mmap = {v: j for j, v in enumerate(self.micros)}
        self.imicro = items.item_microcat_id.map(mmap).to_numpy()
        mc = train.item_microcat_id.map(mmap).to_numpy()
        self.qmicro = sparse.csr_matrix(
            (np.ones(len(train), np.float32), (qr, mc)), shape=(len(qmap), len(mmap))
        )
        self.qmicro = l2normalize(self.qmicro, norm="l1").tocsr()

        # Насколько запрос "локальный": как часто выбирали исполнителя из того же города
        self.qcount = np.bincount(qr, minlength=len(qmap))
        eq = (train.search_location_id == train.item_location_id).to_numpy()
        self.qlocal = np.bincount(qr, weights=eq, minlength=len(qmap)) / self.qcount

        # Слова из заголовков выбранных объявлений. Помогает с синонимами:
        # "изготовление памятников" и "производство памятников"
        ut = train[["item_id", "item_title_raw"]].drop_duplicates("item_id")
        itmap = {v: j for j, v in enumerate(ut.item_id)}
        tm = self.wv.transform(ut.item_title_raw.map(words)).multiply(self.fields[0][1])
        tm = l2normalize(tm).tocsr()
        qt = sparse.csr_matrix(
            (np.ones(len(train), np.float32), (qr, train.item_id.map(itmap))),
            shape=(len(qmap), len(ut)),
        )
        qt = l2normalize(qt, norm="l1")
        self.expansion = (qt @ tm).tocsr()

        # Какие объявления корпуса уже выбирали по каждому тексту
        ci = train.item_id.map(self.idmap)
        keep = ci.notna().to_numpy()
        self.qitem = sparse.csr_matrix(
            (np.ones(keep.sum(), np.float32), (qr[keep], ci[keep].astype(int))),
            shape=(len(qmap), self.n),
        )
        self.popularity = np.log1p(np.asarray(self.qitem.sum(axis=0)).ravel())
        self.qitem = l2normalize(self.qitem, norm="l1").tocsr()

        # География
        self.loc = items.item_location_id.to_numpy()
        self.cat = items.item_category_id.to_numpy()
        self.lat = (
            pd.to_numeric(items.item_latitude, errors="coerce")
            .fillna(0)
            .to_numpy(float)
        )
        self.lon = (
            pd.to_numeric(items.item_longitude, errors="coerce")
            .fillna(0)
            .to_numpy(float)
        )
        coords = (
            pd.DataFrame({"loc": self.loc, "lat": self.lat, "lon": self.lon})
            .groupby("loc")[["lat", "lon"]]
            .median()
        )
        self.centers = {k: tuple(v) for k, v in zip(coords.index, coords.to_numpy())}
        # Некоторых локаций поиска (регионы) нет среди локаций объявлений,
        # для них центр и переходы берем из train
        self.loc_counts = (
            train.groupby(["search_location_id", "item_location_id"]).size().to_dict()
        )
        self.loc_totals = train.groupby("search_location_id").size().to_dict()
        self.search_centers = {}
        for k, g in train.groupby("search_location_id"):
            self.search_centers[k] = (
                pd.to_numeric(g.item_latitude).median(),
                pd.to_numeric(g.item_longitude).median(),
            )

        self.rating = items.item_rating.fillna(0).to_numpy(np.float32)
        self.reviews = np.log1p(
            items.item_rating_reviews_count.fillna(0).to_numpy(np.float32)
        )
        self.price = np.log1p(
            pd.to_numeric(items.item_price, errors="coerce")
            .fillna(0)
            .clip(lower=0)
            .to_numpy(np.float32)
        )
        self.hidden = items.item_is_phone_hidden.to_numpy(np.float32)
        self.forbidden = items.item_is_message_forbidden.to_numpy(np.float32)
        del self.items
        log("Index ready")

    def retrieve(self, row, pool=600, extra_scores=None):
        """Собирает пул кандидатов для одного запроса и считает для них признаки."""
        text = norm(row.search_query)
        wt = words(text)
        q = self.wv.transform([wt]).tocsr()
        scores = []
        for mat, idf in self.fields:
            weights = idf[q.indices]
            s = np.asarray(mat[:, q.indices] @ weights).ravel() / max(
                weights.sum(), 1e-6
            )
            scores.append(s)
        ch = (self.cv.transform([text]) @ self.char).toarray().ravel()

        # 40 самых похожих запросов train, близкие получают больший вес
        sims = (
            0.6 * (self.qcv.transform([text]) @ self.qchar).toarray().ravel()
            + 0.4 * (self.qwv.transform([wt]) @ self.qword).toarray().ravel()
        )
        near = topk(sims, 40)
        sw = sims[near] ** 6
        if sw.sum() > 0:
            sw /= sw.sum()
        sq = sparse.csr_matrix(
            (sw, (np.zeros(len(near), int), near)), shape=(1, len(self.qtexts))
        )
        mp = (sq @ self.qmicro).toarray().ravel()[self.imicro]
        beh = (sq @ self.qitem).toarray().ravel()
        localness = float(sw @ self.qlocal[near])
        ex = (sq @ self.expansion).tocsr()
        if ex.nnz > 32:
            take = topk(ex.data, 32)
            ex = sparse.csr_matrix(
                (ex.data[take], (np.zeros(len(take), int), ex.indices[take])),
                shape=ex.shape,
            )
        expand = (l2normalize(ex) @ self.title_tfidf.T).toarray().ravel()

        location = int(row.search_location_id)
        same = (self.loc == location).astype(np.float32)
        center = self.centers.get(location, self.search_centers.get(location))
        if center:
            # Расстояние по формуле гаверсинуса
            a = (
                np.sin(np.radians(self.lat - center[0]) / 2) ** 2
                + np.cos(np.radians(self.lat))
                * np.cos(np.radians(center[0]))
                * np.sin(np.radians(self.lon - center[1]) / 2) ** 2
            )
            distance = 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        else:
            distance = np.full(self.n, 1000.0)
        geo = np.exp(-distance / 50).astype(np.float32)
        lpmap = {
            loc: self.loc_counts.get((location, loc), 0)
            / max(self.loc_totals.get(location, 0), 1)
            for loc in np.unique(self.loc)
        }
        lp = np.array([lpmap[v] for v in self.loc], dtype=np.float32)

        t, d, p = scores
        lex = 0.43 * t + 0.30 * d + 0.07 * p + 0.35 * ch
        semantic = lex + 0.3 * expand + 0.18 * np.sqrt(mp)
        # Для "локальных" запросов сильно штрафуем другие города, для остальных почти нет
        geofactor = (0.22 + 0.50 * same + 0.28 * geo) * localness + (1 - localness)
        base = semantic * geofactor + 0.05 * beh
        # Берем топы разных каналов, чтобы в пул попадали и исполнители из других городов
        candidates = np.unique(
            np.concatenate(
                [
                    topk(base, pool),
                    topk(lex * geofactor, 100),
                    topk(semantic, 60),
                    topk(expand * geofactor, 80),
                    topk(beh * geofactor, 50),
                    topk((np.sqrt(mp) + 0.03 * self.reviews) * geofactor, 60),
                ]
            )
        )
        if extra_scores is not None:
            # Кандидаты от E5 добавляются к лексическим, а не заменяют их.
            # Косинус сдвигаем на уровень 2000-го объявления, чтобы не зависеть от масштаба модели
            dense = extra_scores['dense']
            dense_score = np.maximum(dense - np.partition(dense, -2000)[-2000], 0)
            dense_score /= max(dense_score.max(), 1e-6)
            candidates = np.unique(np.concatenate([
                candidates, topk(dense, 150),
                topk(dense_score * geofactor, 500),
                topk(dense_score * (0.3 + same), 300),
                topk((dense_score + 0.3 * extra_scores['dense_microcat']) * geofactor, 150),
                topk(semantic / max(semantic.max(), 1e-6) + dense_score, 300),
            ]))

        j = candidates
        exact = np.array([float(text in self.titles[z]) for z in j], np.float32)
        features = np.column_stack(
            [
                t[j],
                d[j],
                p[j],
                ch[j],
                expand[j],
                mp[j],
                beh[j],
                same[j],
                np.log1p(distance[j]),
                geo[j],
                lp[j],
                self.rating[j],
                self.reviews[j],
                self.price[j],
                self.popularity[j],
                self.hidden[j],
                self.forbidden[j],
                # search_category = 0 значит поиск по всем категориям. В бенчмарке
                # таких запросов 9%, а в train почти нет, поэтому считаем, что категория совпала
                (self.cat[j] == row.search_category) | (row.search_category == 0),
                exact,
                base[j],
                lex[j],
                np.full(len(j), localness),
                np.full(len(j), sims[near[0]]),
                np.full(len(j), len(text.split())),
                np.full(len(j), len(text)),
                np.full(len(j), bool(row.search_infm_params_text)),
            ]
        ).astype(np.float32)
        # Фильтры поиска сравниваем с параметрами объявления тем же BM25, без жесткого отсева
        fq = self.wv.transform([words(row.search_infm_params_text)]).tocsr()
        mat, idf = self.fields[2]
        fs = np.asarray(mat[:, fq.indices][j, :] @ idf[fq.indices]).ravel() / max(
            idf[fq.indices].sum(), 1e-6
        )
        features = np.column_stack([features, fs])
        if extra_scores is not None:
            additions = [v[j] if np.ndim(v) else np.full(len(j), v)
                         for v in extra_scores.values()]
            # Какая доля слов запроса есть в поле: отличает частичное совпадение от полного
            terms = q.indices[self.unigram[q.indices]]
            for mat, idf in self.fields:
                covered = mat[:, terms][j, :].tocsr()
                covered.data[:] = 1
                additions.append(np.asarray(covered @ idf[terms]).ravel() / max(idf[terms].sum(), 1e-6))
            query_tfidf = l2normalize(q.multiply(self.fields[0][1])).tocsr()
            additions.append((self.title_tfidf[j] @ query_tfidf.T).toarray().ravel())
            features = np.column_stack([features, *additions]).astype(np.float32)
        return candidates, features


FEATURES = [
    "title_bm25",
    "description_bm25",
    "params_bm25",
    "title_char",
    "expanded_title",
    "microcat_probability",
    "neighbor_item_probability",
    "same_location",
    "log_distance",
    "nearby",
    "location_transition",
    "rating",
    "log_reviews",
    "log_price",
    "train_popularity",
    "phone_hidden",
    "message_forbidden",
    "same_category",
    "query_in_title",
    "base_score",
    "lexical_score",
    "query_localness",
    "nearest_query_similarity",
    "query_words",
    "query_length",
    "has_filters",
    "filter_match",
]


def make_features(model, queries, truth=None):
    """Кандидаты и признаки для всех запросов. Если есть truth, то и метки 0/1."""
    xs, ys, groups, candidates = [], [], [], []
    for k, row in enumerate(queries.itertuples(index=False)):
        ids, x = model.retrieve(row)
        candidates.append(ids)
        xs.append(x)
        if truth is not None:
            positives = set(truth[row.query_id])
            ys.append(np.array([v in positives for v in model.ids[ids]], np.int8))
        groups.extend([k] * len(ids))
        if (k + 1) % 100 == 0:
            log("Queries", k + 1, "/", len(queries))
    return (
        np.vstack(xs),
        np.concatenate(ys) if ys else None,
        np.array(groups),
        candidates,
    )


def evaluate(scores, candidates, queries, truth, ids):
    """Recall@50 по запросам, плюс recall всего пула (потолок для отбора)."""
    recalls = []
    ceilings = []
    offset = 0
    for row, idx in zip(queries.itertuples(index=False), candidates):
        s = scores[offset : offset + len(idx)]
        offset += len(idx)
        target = set(truth[row.query_id])
        pred = set(ids[idx[topk(s, 50)]])
        recalls.append(len(pred & target) / len(target))
        ceilings.append(len(set(ids[idx]) & target) / len(target))
    r = np.array(recalls)
    return {
        "recall_at_50": float(r.mean()),
        "pool_recall": float(np.mean(ceilings)),
        "unseen_recall": float(r[queries.unseen.to_numpy()].mean()),
        "seen_recall": float(r[~queries.unseen.to_numpy()].mean()),
        "queries": len(queries),
    }


def validate_answer(path, queries, items):
    """Проверка answer.csv на все требования из условия."""
    answer = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert answer.columns.tolist() == ["query_id", "answer"]
    assert len(answer) == len(queries) and answer.query_id.is_unique
    assert set(answer.query_id) == set(queries.query_id)
    assert answer.query_id.str.len().eq(16).all()
    valid = set(items.item_id)
    for v in answer.answer:
        tokens = v.split()
        assert len(tokens) <= 50 and len(set(tokens)) == len(tokens)
        assert all(re.fullmatch("[0-9a-f]{16}", s) and s in valid for s in tokens)
    log(
        "CSV verified:",
        len(answer),
        "queries; candidate counts",
        answer.answer.str.split().str.len().value_counts().to_dict(),
    )
