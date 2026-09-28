"""
models_final.py — shared code of 3_model_training_final.ipynb (training) and 4_predictions_final.py (test).
Written by the notebook's first cell — edit it there.  Both files import THIS module, so the
level-2 features and the decision rule are computed by exactly the same code on validation and test.

  * time-budget callbacks (LightGBM / XGBoost / CatBoost stop by the clock)
  * model_proba / kfold_proba   probability of one fold model / the mean of a K-fold bundle
  * load_fields                 integer codes of a few clean-CSV columns (names, address parts)
                                + S1 word sets (sparse) for S1-vs-S1 duplicate similarity
  * level2_matrix               graph features of a probability vector: S1 side, candidate side,
                                similarity to the S1's confident matches, duplicate-S1 evidence
  * apply_rule                  decision rule: threshold + soft one-S1-per-candidate + per-rank
                                thresholds, optionally per country (France etc. -> default rule)
"""
import time

import numpy as np
import pandas as pd

PRED_CHUNK = 500_000

# clean-CSV columns used by the graph features (whatever is missing is simply skipped)
CAND_FIELDS = ["name_compact", "name_skeleton", "name_first_token", "address_core",
               "address_pin", "house_number", "street_name"]
S1_FIELDS = ["name_compact", "name_skeleton", "address_core", "address_pin", "house_number"]
S1_TOKENS = {"name": "name_tokens_sorted", "addr": "address_tokens_sorted"}   # word sets for S1-S1 Jaccard


# ============================================================================
#  time budgets
# ============================================================================
def lgb_time_limit(seconds):
    """LightGBM callback: stop after `seconds`, keeping the best iteration seen so far."""
    import lightgbm as lgb
    st = {"t0": None, "best": np.inf, "it": 0, "res": None}

    def _cb(env):
        if st["t0"] is None:
            st["t0"] = time.time()
        if env.evaluation_result_list:
            r = env.evaluation_result_list[0]
            v = -r[2] if r[3] else r[2]
            if v < st["best"]:
                st["best"], st["it"], st["res"] = v, env.iteration, env.evaluation_result_list
        if time.time() - st["t0"] > seconds:
            raise lgb.callback.EarlyStopException(st["it"], st["res"] or env.evaluation_result_list)
    _cb.order = 40
    return _cb


def xgb_time_limit(seconds):
    import xgboost as xgb

    class _T(xgb.callback.TrainingCallback):
        def __init__(self):
            super().__init__()
            self.t0 = None

        def after_iteration(self, model, epoch, evals_log):
            if self.t0 is None:
                self.t0 = time.time()
            return time.time() - self.t0 > seconds
    return _T()


class CatTimeLimit:
    def __init__(self, seconds):
        self.seconds, self.t0 = seconds, None

    def after_iteration(self, info):
        if self.t0 is None:
            self.t0 = time.time()
        return time.time() - self.t0 < self.seconds      # True = keep going


# ============================================================================
#  probabilities
# ============================================================================
def model_proba(m, X):
    """m = {"lib": lightgbm|xgboost|catboost|sklearn, "model": ..., "iters": n, "fill": v}"""
    lib, mod = m["lib"], m["model"]
    out = np.empty(len(X), np.float32)
    for lo in range(0, len(X), PRED_CHUNK):
        xb = np.asarray(X[lo:lo + PRED_CHUNK], dtype=np.float32)
        if m.get("fill") is not None:
            xb = np.where(np.isnan(xb), np.float32(m["fill"]), xb)
        if lib == "lightgbm":
            out[lo:lo + len(xb)] = mod.predict(xb, num_iteration=m.get("iters") or None)
        elif lib == "xgboost":
            import xgboost as xgb
            it = m.get("iters")
            out[lo:lo + len(xb)] = mod.predict(xgb.DMatrix(xb, missing=np.nan),
                                               iteration_range=(0, it) if it else (0, 0))
        else:
            out[lo:lo + len(xb)] = mod.predict_proba(xb)[:, 1]
    return out


def rows_proba(m, X, rows):
    """model_proba on X[rows] without copying all of X[rows] at once."""
    out = np.empty(len(rows), np.float32)
    for lo in range(0, len(rows), PRED_CHUNK):
        out[lo:lo + PRED_CHUNK] = model_proba(m, X[rows[lo:lo + PRED_CHUNK]])
    return out


def kfold_proba(bundle, X):
    """test / validation probability of a K-fold bundle = mean of its fold models."""
    return np.mean([model_proba(m, X) for m in bundle["models"]], axis=0).astype(np.float32)


# ============================================================================
#  group helpers
# ============================================================================
def group_max(v, g, n=None):
    n = int(g.max()) + 1 if n is None else n
    m = np.full(n, -np.inf)
    np.maximum.at(m, g, v)
    return m[g]


def rank_lead(g, v):
    """rank of each row inside its group (0 = highest v) and v minus the best OTHER value
    of the group (NaN for a group of one)."""
    n = len(v)
    vf = np.where(np.isnan(v), -np.inf, v).astype(np.float64)
    o = np.lexsort((-vf, g))
    go = g[o]
    st = np.flatnonzero(np.concatenate(([True], go[1:] != go[:-1])))
    gl = np.diff(np.append(st, n))
    first = np.repeat(st, gl)
    rank = np.empty(n, np.float32)
    rank[o] = np.arange(n) - first
    vo = vf[o]
    best = np.repeat(vo[st], gl)
    sec = np.repeat(np.where(gl >= 2, vo[np.minimum(st + 1, n - 1)], -np.inf), gl)
    other = np.empty(n, np.float64)
    other[o] = np.where(np.arange(n) == first, sec, best)
    with np.errstate(invalid="ignore"):
        lead = (vf - other).astype(np.float32)
    lead[~np.isfinite(other) | ~np.isfinite(vf)] = np.nan
    return rank, lead


def _top2_other(g, v, s):
    """for every row: the `s` value of the best OTHER row of its group `g` (by v), -1 if none."""
    n = len(v)
    o = np.lexsort((-v, g))
    go = g[o]
    st = np.flatnonzero(np.concatenate(([True], go[1:] != go[:-1])))
    gl = np.diff(np.append(st, n))
    so = s[o]
    t1 = so[st]
    t2 = np.where(gl >= 2, so[np.minimum(st + 1, n - 1)], -1)
    first = np.repeat(st, gl)
    is_first = np.empty(n, bool)
    is_first[o] = np.arange(n) == first
    gid = np.empty(n, np.int64)
    gid[o] = np.repeat(np.arange(len(st)), gl)
    return np.where(is_first, t2[gid], t1[gid])


def _lookup(keys_sorted, vals, q):
    """vals[k] where keys_sorted[k] == q, else 0."""
    if len(keys_sorted) == 0:
        return np.zeros(len(q), np.float64)
    pos = np.minimum(np.searchsorted(keys_sorted, q), len(keys_sorted) - 1)
    return np.where(keys_sorted[pos] == q, vals[pos], 0).astype(np.float64)


# ============================================================================
#  clean-CSV field codes
# ============================================================================
def load_fields(art, split, s1_cats, cand_cats):
    """-> (s1 fields, candidate fields): {column: int32 code per category}, -1 = empty.
    Codes are only compared within the same table side, so each column is factorised alone."""
    from pathlib import Path
    art = Path(art)

    def read(name, fields):
        f = art / f"{split}_{name}_clean.csv"
        if not f.exists():
            return None
        have = pd.read_csv(f, nrows=0).columns
        use = ["entity_id"] + [c for c in fields if c in have]
        return pd.read_csv(f, usecols=use, dtype=str, keep_default_na=False)

    def codes(df, cats, fields):
        out = {}
        if df is None:
            return out
        df = df.drop_duplicates("entity_id").set_index("entity_id")
        for c in fields:
            if c not in df:
                continue
            v = df[c].reindex(pd.Index(cats)).fillna("").astype(str).str.strip().to_numpy(dtype=object)
            k, _ = pd.factorize(v)
            k = k.astype(np.int32)
            k[v == ""] = -1
            out[c] = k
        return out

    s1 = read("s1", S1_FIELDS + list(S1_TOKENS.values()))
    cd = [d for d in (read("s2", CAND_FIELDS), read("s3", CAND_FIELDS)) if d is not None]
    cd = pd.concat(cd, ignore_index=True) if cd else None
    sf = codes(s1, s1_cats, S1_FIELDS)
    if s1 is not None:
        s1i = s1.drop_duplicates("entity_id").set_index("entity_id")
        for key, colname in S1_TOKENS.items():
            if colname in s1i:
                m = _tokens_csr(s1i[colname].reindex(pd.Index(s1_cats)).fillna("").astype(str).to_numpy(dtype=object))
                if m is not None:
                    sf[f"_tok_{key}"] = m
    return sf, codes(cd, cand_cats, CAND_FIELDS)


def _tokens_csr(values):
    """one row per value, a 1 for every distinct word (scipy CSR); None without scipy."""
    try:
        from scipy.sparse import csr_matrix
    except Exception:
        return None
    t = pd.Series(values, index=np.arange(len(values))).str.split().explode()
    t = t[t.notna() & (t != "")]
    n = len(values)
    if not len(t):
        return csr_matrix((n, 1), dtype=np.float32)
    code, _ = pd.factorize(t.to_numpy())
    m = csr_matrix((np.ones(len(code), np.float32), (t.index.to_numpy(), code)), shape=(n, int(code.max()) + 1))
    m.sum_duplicates()
    m.data[:] = 1.0
    return m


def _pair_jaccard(M, a, b, chunk=2_000_000):
    """Jaccard of the word sets of rows a[i] and b[i] of M (NaN when both are empty)."""
    ns = M.shape[0]
    key = a.astype(np.int64) * ns + b.astype(np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    out = np.full(len(uk), np.nan)
    for lo in range(0, len(uk), chunk):
        k = uk[lo:lo + chunk]
        A, B = M[k // ns], M[k % ns]
        inter = np.asarray(A.multiply(B).sum(axis=1)).ravel()
        uni = np.diff(A.indptr) + np.diff(B.indptr) - inter
        out[lo:lo + chunk] = np.where(uni > 0, inter / np.maximum(uni, 1), np.nan)
    return out[inv]


# ============================================================================
#  level-2 (graph) features
# ============================================================================
def level2_names(extra_names, cf_fields, sf_fields, raw_names, tok_keys=()):
    n = ["p"] + [f"x_{e}" for e in extra_names] + (["x_std"] if len(extra_names) >= 2 else [])
    n += ["s1_rank", "s1_lead", "p_div_s1max", "s1_sum", "s1_n50", "s1_n90", "s1_npairs", "rank_minus_n50"]
    n += ["c_rank", "c_lead", "p_div_cmax", "c_sum", "c_n50", "c_n90", "c_npairs"]
    n += ["anchor_n_other"] + [f"anchor_share_{f}" for f in cf_fields] + ["anchor_share_max"]
    n += ["other_s1_p", "other_s1_shared", "other_s1_shared_frac", "s1_dup_shared_max"]
    n += [f"other_s1_same_{f}" for f in sf_fields] + [f"other_s1_jacc_{t}" for t in tok_keys]
    n += [f"raw_{r}" for r in raw_names]
    return n


def level2_matrix(p, extras, raw, raw_names, s1, cand, sf, cf, anchor_p=0.9, conf_p=0.5, max_share=50):
    """p: main probability (all rows of the split); extras: list of (name, prob vector);
    raw: (n, k) raw pair features; s1/cand: integer codes; sf/cf: field codes per S1 / candidate code.
    Returns (Z float32 (n, d), names).  Everything is computed over ALL rows given — candidates
    compete across every S1 they appear with (validation and test alike)."""
    p = np.asarray(p, np.float64)
    n = len(p)
    cff = sorted(cf)
    sff = sorted(k for k in sf if not k.startswith("_"))
    toks = sorted(k[5:] for k in sf if k.startswith("_tok_"))
    names = level2_names([e for e, _ in extras], cff, sff, raw_names, toks)
    Z = np.full((n, len(names)), np.nan, np.float32)
    col = {c: j for j, c in enumerate(names)}

    def put(name, v):
        Z[:, col[name]] = v

    put("p", p)
    if extras:
        E = np.column_stack([np.asarray(v, np.float32) for _, v in extras])
        for j, (e, _) in enumerate(extras):
            put(f"x_{e}", E[:, j])
        if len(extras) >= 2:
            put("x_std", E.std(axis=1))
        del E

    # ---- S1 side ----
    ns = int(s1.max()) + 1
    r1, l1 = rank_lead(s1, p)
    mx = group_max(p, s1, ns)
    n50 = np.bincount(s1, weights=(p >= 0.5), minlength=ns)
    n90 = np.bincount(s1, weights=(p >= anchor_p), minlength=ns)
    put("s1_rank", r1); put("s1_lead", l1); put("p_div_s1max", p / np.maximum(mx, 1e-6))
    put("s1_sum", np.bincount(s1, weights=p, minlength=ns)[s1]); put("s1_n50", n50[s1]); put("s1_n90", n90[s1])
    put("s1_npairs", np.bincount(s1, minlength=ns)[s1]); put("rank_minus_n50", r1 - n50[s1])
    del mx

    # ---- candidate side ----
    nc = int(cand.max()) + 1
    rc, lc = rank_lead(cand, p)
    cmx = group_max(p, cand, nc)
    put("c_rank", rc); put("c_lead", lc); put("p_div_cmax", p / np.maximum(cmx, 1e-6))
    put("c_sum", np.bincount(cand, weights=p, minlength=nc)[cand])
    put("c_n50", np.bincount(cand, weights=(p >= 0.5), minlength=nc)[cand])
    put("c_n90", np.bincount(cand, weights=(p >= anchor_p), minlength=nc)[cand])
    put("c_npairs", np.bincount(cand, minlength=nc)[cand])
    put("other_s1_p", p - lc)                       # best OTHER S1's p for this candidate (NaN: none)
    del rc, lc, cmx

    # ---- similarity to the S1's confident matches (anchors) ----
    is_a = p >= anchor_p
    n_other = n90[s1] - is_a
    put("anchor_n_other", n_other)
    shares = []
    for f in cff:
        code = cf[f][cand].astype(np.int64)
        ok = code >= 0
        key = s1.astype(np.int64) * (int(code.max()) + 2) + code
        uk, cnt = np.unique(key[is_a & ok], return_counts=True)
        c = _lookup(uk, cnt, key) - (is_a & ok)
        sh = np.where(ok & (n_other > 0), c / np.maximum(n_other, 1), np.nan)
        put(f"anchor_share_{f}", sh)
        shares.append(sh)
    if shares:
        S = np.column_stack(shares)
        allnan = np.isnan(S).all(axis=1)
        put("anchor_share_max", np.where(allnan, np.nan, np.nanmax(np.where(np.isnan(S), -1, S), axis=1)))
        del S

    # ---- the candidate's best OTHER S1: how strong, and is it a duplicate of this S1? ----
    other = _top2_other(cand, p, s1.astype(np.int64))
    has_o = other >= 0
    o_safe = np.where(has_o, other, 0)
    conf = p >= conf_p
    cc = cand[conf]
    cnt = np.bincount(cc, minlength=nc)
    keep = (cnt[cc] >= 2) & (cnt[cc] <= max_share)
    shared = np.zeros(n)
    dupmax = np.zeros(ns)
    if keep.any():
        d = pd.DataFrame({"s": s1[conf][keep].astype(np.int64), "c": cc[keep]})
        m = d.merge(d, on="c")
        m = m[m["s_x"] != m["s_y"]]
        if len(m):
            uk, kc = np.unique(m["s_x"].to_numpy() * ns + m["s_y"].to_numpy(), return_counts=True)
            shared = _lookup(uk, kc, s1.astype(np.int64) * ns + o_safe)
            np.maximum.at(dupmax, uk // ns, kc)            # most likely candidates shared with ANY other S1
        del d, m
    put("other_s1_shared", np.where(has_o, shared, np.nan))
    put("other_s1_shared_frac", np.where(has_o, shared / np.maximum(n50[s1], 1), np.nan))
    put("s1_dup_shared_max", dupmax[s1])
    for f in sff:
        a, b = sf[f][s1], sf[f][o_safe]
        put(f"other_s1_same_{f}", np.where(has_o & (a >= 0) & (b >= 0), (a == b).astype(np.float32), np.nan))
    for t in toks:                                          # word overlap of this S1 and the other S1
        v = np.full(n, np.nan)
        if has_o.any():
            idx = np.flatnonzero(has_o)
            v[idx] = _pair_jaccard(sf[f"_tok_{t}"], s1[idx], other[idx])
        put(f"other_s1_jacc_{t}", v)

    # ---- raw pair features ----
    for j, r in enumerate(raw_names):
        put(f"raw_{r}", raw[:, j])
    return Z, names


# ============================================================================
#  decision rule
# ============================================================================
def _decide(p, s1, cand, r, cmax):
    """match if p >= thr and p >= opc_r * (the candidate's best p over all its S1s)
    [opc_r = 1: a candidate goes only to its best S1; < 1: also to S1s almost as good; 0: off],
    OR the pair is ranked k-th among its S1's surviving pairs and p >= thr_ranks[k]."""
    base = p >= float(r.get("opc_r", 1.0)) * cmax - 1e-12
    pred = base & (p >= float(r["thr"]))
    ranks = [(int(k), float(t)) for k, t in (r.get("thr_ranks") or []) if float(t) < float(r["thr"])]
    if ranks:
        rank, _ = rank_lead(s1, np.where(base, p, -np.inf))
        for k, t in ranks:
            pred |= base & (rank == k) & (p >= t)
    return pred


def apply_rule(p, s1, cand, rule, country=None, cmax=None):
    """rule = {"default": {thr, opc_r, thr_ranks}, "by_country": {country: {...}}}.
    cmax = each row's candidate-best p (computed over all rows given when None)."""
    p = np.asarray(p, np.float64)
    if cmax is None:
        cmax = group_max(p, cand)
    by = rule.get("by_country") or {}
    if country is None or not by:
        return _decide(p, s1, cand, rule["default"], cmax)
    pred = np.zeros(len(p), bool)
    country = np.asarray(country, dtype=object)
    for c in pd.unique(country):
        m = country == c
        pred[m] = _decide(p[m], s1[m], cand[m], by.get(c, rule["default"]), cmax[m])
    return pred


def round_inputs(rnd, l1, prev=None):
    """inputs of graph round `rnd` (1-based): (main probability, extra probability columns).
    l1 = {model name: probability} in a fixed order; prev = the previous round's probability."""
    names = list(l1)
    p1 = np.mean([np.asarray(l1[m], np.float64) for m in names], axis=0)
    ex = [(m, l1[m]) for m in names] if len(names) >= 2 else []
    if rnd == 1:
        return p1, ex
    return prev, ex + [("l1_mean" if len(names) >= 2 else "l1", p1)]
