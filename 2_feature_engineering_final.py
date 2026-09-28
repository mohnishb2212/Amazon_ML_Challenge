"""
2_feature_engineering_final.py — Stage 2: pair features for the matching model.   (final)

    input :  artifacts/candidate_pairs_{split}.tsv        (from 1_blocking_final.py)
             artifacts/{split}_s1/s2/s3_clean.csv          (from 0_eda_preprocessing_final.ipynb)
             artifacts/train_val_split.csv                 (optional, train only)
    output:  artifacts/pair_features_{split}.parquet       (falls back to .tsv
                                                            if pyarrow is missing)

Run:   python 2_feature_engineering_final.py            -> train
       python 2_feature_engineering_final.py test       -> test
       python 2_feature_engineering_final.py all        -> train, then test

One row per (source1_entity_id, candidate_entity_id).  Feature groups:

  NAME          edit distance (Levenshtein, Jaro-Winkler, token sort / set, partial,
                compact), Jaccard / overlap, word + char-trigram TF-IDF, Soft-TFIDF,
                Monge-Elkan, IDF-weighted soft coverage, leftover words (real vs
                candidate-only noise words), rarity, sound-alike keys
  NAME+ (new)   information shared (sum of IDF of shared words), candidate-name
                rarity, numbers inside the names (874 vs 875), acronym / initials
                match (lhs = lucky hair studio, dicarecom = dental interstate care ...)
  ALIAS (new)   DBA / formerly / aka halves from preprocessing: best agreement over
                every (name, primary, alias) x (name, primary, alias) combination
  LEGAL         legal form present / equal / Jaccard
  ADDRESS       whole-address similarities, soft street words, numbers, postal code
  FIELDS (new)  the address split into fields by preprocessing: house number, unit /
                suite, floor, plot / sector / phase / block (agree / conflict counts),
                street name, landmark, exact address
  CROSS (new)   name words found in the other record's address (a business named
                after its street / area / building)
  NO-ADDRESS    name evidence when an address is missing
  CONTEXT       group sizes, gaps, ranks and leads (blocking probability, name, address,
                and a combined name+address score), mutual best pair, how many other
                S1 / candidates in play carry the SAME name (live ambiguity), first-word
                frequency, how many S1 share the candidate's exact address (multi-tenant
                buildings), candidate source (S2 / S3, from the file it came from)
  SIBLING (new) agreement between the candidate and the S1's best OTHER candidate
                (candidates of one entity are near-duplicates of each other, so a
                candidate with no address can still borrow its sibling's evidence)
  BLOCKING      score, n_keys, n_evidence, cand_rank, p_block and one bit per
                blocking key family (read from 1_blocking_final.py, any number)

NULL POLICY
  * Missing text is never turned into a fake value. A similarity that cannot be
    computed (field missing on either side) is NaN — GBDTs learn a branch for it,
    so "no data" is never confused with "totally different".
  * Explicit 0/1 flags / counts carry the missingness itself.

COUNTRY-AGNOSTIC BY CONSTRUCTION
  * Nothing is keyed on a country name; the script loops over whatever country
    values appear. Vocabularies, IDF, noise words and counts are fitted per country
    on that split's own records (no labels touched).
  * Country is written as a metadata column only — do NOT feed it to the model.
"""
import gc
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler, Levenshtein
    from rapidfuzz.process import cpdist
except ImportError:  # pragma: no cover
    raise SystemExit("rapidfuzz >= 3.6 is required:  pip install 'rapidfuzz>=3.6'")

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAVE_ARROW = True
except ImportError:  # pragma: no cover
    HAVE_ARROW = False

try:
    BASE = Path(__file__).resolve().parent
except NameError:
    BASE = Path(".").resolve()
ART = BASE / "artifacts"
sys.path.insert(0, str(BASE))

# legal-form canonicalisation comes from the preprocessing code (same rules)
try:
    from prep_final import SUFFIX_MAP, phonetic_skeleton
except Exception:  # pragma: no cover
    try:
        from utils import SUFFIX_MAP, phonetic_skeleton
    except Exception:
        SUFFIX_MAP = {"incorporated": "inc", "inc": "inc", "corporation": "corp", "corp": "corp",
                      "company": "co", "co": "co", "limited": "ltd", "ltd": "ltd",
                      "private": "pvt", "pvt": "pvt", "llc": "llc", "llp": "llp", "pllc": "pllc",
                      "lp": "lp", "sarl": "sarl", "sas": "sas", "sa": "sa"}

        _SKEL_DROP = set("aeiouhy")
        _SKEL_MAP = {**{c: "b" for c in "bvwpf"}, **{c: "s" for c in "cszjx"},
                     **{c: "d" for c in "dt"}, **{c: "k" for c in "kgq"},
                     **{c: c for c in "mnrl"}}

        def phonetic_skeleton(token):
            return "".join(_SKEL_MAP.get(ch, ch if ch.isdigit() else "")
                           for ch in token if ch not in _SKEL_DROP)


def _load_fam():
    """blocking key-family names (bit i of evidence_mask = family i), read from the
    FAMILIES dict of the blocking script WITHOUT executing it."""
    import ast
    for fn in ("1_blocking_final.py", "blocking_final.py", "blocking.py"):
        f = BASE / fn
        if not f.exists():
            continue
        try:
            tree = ast.parse(f.read_text())
        except Exception:
            continue
        for node in tree.body:
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
                    and any(isinstance(t, ast.Name) and t.id == "FAMILIES" for t in node.targets)):
                keys = [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
                if keys:
                    return keys, fn
    return [f"{i:02d}" for i in range(24)], None


BLK_FAM, BLK_SRC = _load_fam()

# ============================================================================
#  knobs
# ============================================================================
READ_CHUNK = 500_000      # CSV rows per read
PAIR_CHUNK = 500_000      # pairs per feature batch (memory bound)
WRITE_CHUNK = 1_000_000   # rows per parquet write
ROW_CAP = 4_000_000       # word x word rows per soft-matching sub-batch (memory bound)
WORKERS = -1              # rapidfuzz threads (-1 = all cores)
CHAR_NGRAM = (3, 3)       # char n-grams for name TF-IDF
REPORT_SAMPLE = 500_000   # rows sampled for the train feature report
CORR_WARN = 0.97          # report feature pairs more correlated than this

SOFT_THETA = 0.90         # Jaro-Winkler at which two words count as the same word
K_NAME = 6                # rarest words per name used by soft matching
K_ADDR = 5                # rarest street / locality words per address
K_NUM = 4                 # rarest numbers per address
NOISE_DF = 50             # candidate-only noise word: in >= this many candidate names ...
NOISE_RATIO = 3.0         # ... and this many times more frequent than among S1 names
STRONG_P = 0.5            # s1_n_strong: pairs with p_block above this

FILES = {
    "train": dict(s1="train_s1_clean.csv",
                  cd=["train_s2_clean.csv", "train_s3_clean.csv"]),
    "test":  dict(s1="test_s1_clean.csv",
                  cd=["test_s2_clean.csv", "test_s3_clean.csv"]),
}
REC_COLS = ["entity_id", "country_clean", "business_name", "name_translit",
            "name_core", "address_clean", "address_pin", "is_missing_address",
            # structured columns from preprocessing (used when present)
            "name_primary", "name_alias",
            "house_number", "unit_number", "floor_number", "plot_number",
            "sector", "phase", "block", "street_name", "landmark"]
SLOT_COLS = ["house_number", "unit_number", "floor_number", "plot_number",
             "sector", "phase", "block"]
SUBLOC_COLS = ["plot_number", "sector", "phase", "block"]
ABBR_MAX = 12             # acronym side: one word of at most this many letters

NAME_FEATS = [
    "name_lev", "name_jw", "name_token_sort", "name_token_set",
    "name_partial_compact", "name_compact_lev", "name_char_tfidf_cos",
    "name_word_tfidf_cos", "name_jaccard", "name_overlap",
    "name_idf_shared_max", "name_idf_unshared_s1", "name_idf_unshared_cand",
    "name_phon_lev", "name_phon_jaccard", "name_first_tok_eq",
    "name_len_ratio", "name_script_mismatch", "name_is_web",
    # soft token matching
    "name_soft_tfidf", "name_me_s1", "name_me_cand", "name_softcov_s1", "name_softcov_cand",
    "name_unm_s1", "name_unm_cand_real", "name_unm_cand_noise",
    "name_unm_idf_s1", "name_unm_idf_cand",
    "name_set_eq", "name_denoised_eq", "name_rarity_s1", "name_ntok_s1", "name_ntok_cand",
    "phon_soft_tfidf", "phon_softcov_s1", "phon_softcov_cand",
]
LEGAL_FEATS = ["lf_s1", "lf_cand", "lf_eq", "lf_jacc"]
ADDR_FEATS = [
    "addr_missing_s1", "addr_missing_cand", "addr_token_set", "addr_sorted_lev",
    "addr_word_tfidf_cos", "addr_jaccard", "addr_overlap",
    "addr_idf_shared_max", "addr_idf_unshared_cand", "addr_len_ratio",
    "addr_num_jaccard", "addr_num_overlap", "addr_first_num_eq",
    "pin_eq", "pin_prefix_eq",
    # soft street / locality matching + numbers + postal code
    "addr_soft_tfidf", "addr_softcov_s1", "addr_softcov_cand", "addr_me_cand",
    "addr_unm_idf_cand", "addr_num_best_lev", "addr_num_best_close",
    "addr_hnum_lev", "addr_hnum_close", "addr_street_same_num_diff", "pin_lev",
]
INTER_FEATS = ["noaddr_name_soft", "noaddr_name_eq"]
# ---- new in the final version ----
NAMEX_FEATS = ["name_idf_shared_sum", "name_rarity_cand", "name_num_eq", "name_num_one_side",
               "name_abbrev"]
ALIAS_FEATS = ["alias_any", "alias_best_tset", "alias_best_jw", "alias_exact", "alias_gain"]
FIELD_FEATS = ["addr_exact_eq", "addr_idf_shared_sum",
               "f_house_eq", "f_unit_eq", "f_subloc_eq",
               "fld_n_both", "fld_n_eq", "fld_n_conflict",
               "f_street_tset", "f_street_jw", "f_landmark_tset"]
CROSS_FEATS = ["x_s1name_in_candaddr", "x_candname_in_s1addr"]
CTX_FEATS = [
    "s1_n_cand", "cand_n_s1", "name_gap_cand", "addr_gap_cand",
    "name_gap_s1", "addr_gap_s1", "s1_name_dup", "cand_name_ambig",
    "pb_rank_s1", "pb_vs_other_s1", "pb_vs_other_cand",
    "s1_pb_sum", "s1_n_strong", "cand_pb_sum",
    "nsoft_rank_s1", "nsoft_vs_other_s1", "nsoft_vs_other_cand",
    "asoft_vs_other_s1", "asoft_vs_other_cand",
    # new
    "mutual_best_pb", "combo", "combo_rank_s1", "combo_vs_other_s1", "combo_vs_other_cand",
    "mutual_best_combo", "s1_n_cand_same_name", "cand_n_s1_same_name",
    "cand_fw_s1df", "name_cd_cnt_cand", "cand_addr_s1_cnt", "s1_addr_dup",
    "cand_src", "pb_rank_s1_src",
]
SIB_FEATS = ["sib_anchor_pb", "sib_name_tset", "sib_addr_tset", "sib_pin_eq"]
BLK_MAP = {"score": "blk_score", "n_keys": "blk_n_keys",
           "n_evidence": "blk_n_evidence", "cand_rank": "blk_cand_rank",
           "p_block": "blk_p_block"}
EVB_FEATS = [f"evb_{f}" for f in BLK_FAM]
BLK_FEATS = list(BLK_MAP.values()) + EVB_FEATS
FEATURES = (NAME_FEATS + NAMEX_FEATS + ALIAS_FEATS + LEGAL_FEATS + ADDR_FEATS + FIELD_FEATS
            + CROSS_FEATS + INTER_FEATS + CTX_FEATS + SIB_FEATS + BLK_FEATS)
FIDX = {f: i for i, f in enumerate(FEATURES)}

# name similarities that are meaningless when either name is empty
_NAME_SIM = NAME_FEATS[:17] + ["name_soft_tfidf", "name_me_s1", "name_me_cand",
                                "name_softcov_s1", "name_softcov_cand", "name_unm_s1",
                                "name_unm_cand_real", "name_unm_cand_noise", "name_unm_idf_s1",
                                "name_unm_idf_cand", "name_set_eq", "name_denoised_eq"]
# address similarities that are meaningless when either address is missing
_ADDR_SIM = ADDR_FEATS[2:13] + ["addr_soft_tfidf", "addr_softcov_s1", "addr_softcov_cand",
                                "addr_me_cand", "addr_unm_idf_cand", "addr_num_best_lev",
                                "addr_num_best_close", "addr_hnum_lev", "addr_hnum_close",
                                "addr_street_same_num_diff"]


# ============================================================================
#  text normalisation (pure functions, applied once per UNIQUE string)
# ============================================================================
_WEB_RE = re.compile(r"(?i)(?:www\.|\.(?:com|net|org|info|biz|io|co|in|fr|us|uk)\b|^\s*[#@])")
_TLD = ("info", "com", "net", "org", "biz", "io", "co", "in", "fr", "us", "uk")
_DIG_FIX = str.maketrans("013458", "oleasb")          # look-alike digits
_ONE_DIGIT = re.compile(r"(?<![0-9])[013458](?![0-9])")
_NON_AZ = re.compile(r"[^a-z]")
_SOFT_C = re.compile(r"c(?=[eiy])")
_NASAL = re.compile(r"m(?=[^aeioupbmv])")               # anusvara-style m -> n
_DOUBLE = re.compile(r"(.)\1+")
_VOWELS = frozenset("aeiou")
_NULL_TOK = frozenset({"null", "none", "nan", "nil", "na"})
_WORD_ORD = {"first": "1", "second": "2", "third": "3", "fourth": "4",
             "fifth": "5", "sixth": "6", "seventh": "7", "eighth": "8",
             "ninth": "9", "tenth": "10", "eleventh": "11", "twelfth": "12"}
_NUM_TOK = re.compile(r"(?:no)?0*(\d+)(?:st|nd|rd|th)?")


def _fix_tok(t):
    """sik0ra -> sikora, 8eia -> beia; leaves real numbers (221health, 3m) alone."""
    t = t.replace("'", "")
    if not t or t.isalpha() or t.isdigit():
        return t
    if sum(c.isalpha() for c in t) < 2:
        return t
    return _ONE_DIGIT.sub(lambda m: m.group(0).translate(_DIG_FIX), t)


def _strip_web(toks):
    """www.edgeland.com -> edgeland  (only for names that look like a URL/handle)."""
    if toks and toks[0] == "www":
        toks = toks[1:]
    if not toks:
        return toks
    last = toks[-1]
    if last in _TLD and len(toks) > 1:
        return toks[:-1]
    for s in _TLD:
        if last.endswith(s) and len(last) - len(s) >= 3:
            return toks[:-1] + [last[:-len(s)]]
    return toks


def _norm_name(key):
    web = key.endswith("\x00")
    s = key[:-1] if web else key
    toks = [t for t in (_fix_tok(x) for x in s.split()) if t]
    if web:
        toks = _strip_web(toks)
    return " ".join(toks)


def _phon_tok(t):
    """Consonant skeleton that survives transliteration and spelling variants:
    aspirates dropped (kh->k, sh->s), soft/hard c, z->s, w->v, m->n before a
    non-labial, vowels dropped after the first letter, doubles collapsed.
    laxmi / lakshmi / laksmi -> lksm ;  shree / sri / shri -> sr"""
    t = _NON_AZ.sub("", t)
    if not t:
        return ""
    t = t.replace("ph", "f")
    t = "".join(ch for i, ch in enumerate(t)
                if not (ch == "h" and i > 0 and t[i - 1] not in _VOWELS))
    if not t:
        return ""
    t = _SOFT_C.sub("s", t)
    t = (t.replace("c", "k").replace("q", "k").replace("x", "ks")
          .replace("z", "s").replace("w", "v").replace("y", "i"))
    t = _NASAL.sub("n", t)
    head = "a" if t[0] in _VOWELS else t[0]
    body = "".join(ch for ch in t[1:] if ch not in _VOWELS)
    return _DOUBLE.sub(r"\1", head + body)


def _phon_name(s):
    return " ".join(p for p in (_phon_tok(t) for t in s.split()) if p)


def _addr_tok(t):
    """0064 -> 64, 316th -> 316, no21 -> 21, fifth -> 5, null -> (dropped)."""
    if t in _NULL_TOK:
        return ""
    if t in _WORD_ORD:
        return _WORD_ORD[t]
    m = _NUM_TOK.fullmatch(t)
    return m.group(1) if m else t


def _norm_addr(s):
    return " ".join(x for x in (_addr_tok(t) for t in s.split()) if x)


def _first_num(s):
    for t in s.split():
        if t.isdigit():
            return t
    return ""


def _derive(raw, fn):
    """Apply fn once per unique value of raw (object array); return per-row result."""
    codes, uni = pd.factorize(raw)
    out = np.array([fn(x) for x in uni], dtype=object)
    return out[codes] if len(codes) else np.array([], dtype=object)


def _codes(values, empty_is_missing=True):
    """factorize, with '' -> -1 so that 'both missing' never counts as a match."""
    codes, _ = pd.factorize(values)
    if empty_is_missing:
        codes = np.where(values == "", -1, codes)
    return codes.astype(np.int64)


# ---------------------------------------------------------------- legal form
_SKEL_TO_CANON = {}
for _w, _c in SUFFIX_MAP.items():
    try:
        _k = phonetic_skeleton(_w)
    except Exception:  # pragma: no cover
        _k = _w
    if len(_k) >= 3:
        _SKEL_TO_CANON.setdefault(_k, _c)
_CANON_MEMO = {}


def _canon_legal(w):
    """'private' / 'pvt' / 'praiveta' -> 'pvt' (same rules as preprocessing)."""
    c = _CANON_MEMO.get(w)
    if c is None:
        c = SUFFIX_MAP.get(w)
        if c is None:
            try:
                c = _SKEL_TO_CANON.get(phonetic_skeleton(w), w)
            except Exception:  # pragma: no cover
                c = w
        _CANON_MEMO[w] = c
    return c


def _legal_masks(trl, core):
    """per record: bitmask of its canonical legal-form words (0 = none).
    legal-form words = words of name_translit that name_core does not have."""
    key = pd.Series(trl, dtype=object) + "\x01" + pd.Series(core, dtype=object)
    codes, uni = pd.factorize(key)
    bit_of = {}
    out = np.zeros(len(uni), np.uint64)
    for u, k in enumerate(uni):
        t, c = k.split("\x01", 1)
        if not t or t == c:
            continue
        cs = set(c.split())
        m = 0
        for w in t.split():
            if w in cs or not w.isalpha():
                continue
            cw = _canon_legal(w)
            b = bit_of.get(cw)
            if b is None:
                b = min(len(bit_of), 62)
                bit_of[cw] = b
            m |= 1 << b
        out[u] = m
    return out[codes] if len(codes) else np.zeros(0, np.uint64)


def _popcount(x):
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.float32)
    x = np.ascontiguousarray(x, dtype=np.uint64)
    t8 = np.array([bin(i).count("1") for i in range(256)], np.uint8)
    return t8[x.view(np.uint8)].reshape(-1, 8).sum(axis=1).astype(np.float32)


# ============================================================================
#  record loading (one country at a time, reduced columns only)
# ============================================================================
_LEAD0 = re.compile(r"(?<![0-9])0+(?=[0-9])")
_NOSPACE = re.compile(r"[\s\-/.,#]+")


def _slot_norm(x):
    """field value -> comparable key: 'No. 012' -> '12', 'B-2' -> 'b2'."""
    x = x.lower()
    for w in ("no", "number", "num", "#"):
        if x.startswith(w + " "):
            x = x[len(w) + 1:]
    return _LEAD0.sub("", _NOSPACE.sub("", x))


def _reduce(ch, src):
    def g(c):
        return ch[c].astype(str) if c in ch.columns else pd.Series("", index=ch.index)
    raw = g("business_name")
    core, trl = g("name_core").str.strip(), g("name_translit").str.strip()
    name = core.where(core != "", trl)
    name = name.where(name != "", raw.str.lower().str.strip())
    out = {
        "id": ch["entity_id"].to_numpy(),
        "name": name.to_numpy(),
        "trl": trl.str.lower().to_numpy(),
        "core": core.str.lower().to_numpy(),
        "addr": g("address_clean").str.strip().to_numpy(),
        "pin": g("address_pin").str.replace(r"\D", "", regex=True).to_numpy(),
        "miss": g("is_missing_address").str.strip().str.lower()
                  .isin(["true", "1", "yes"]).to_numpy(),
        "nonascii": (~raw.map(str.isascii)).to_numpy(dtype=bool),
        "web": raw.str.contains(_WEB_RE, na=False).to_numpy(dtype=bool),
        "src": np.full(len(ch), src, np.int8),
        "prim": g("name_primary").str.strip().str.lower().to_numpy(),
        "alias": g("name_alias").str.strip().str.lower().to_numpy(),
        "street": g("street_name").str.strip().str.lower().to_numpy(),
        "landmark": g("landmark").str.strip().str.lower().to_numpy(),
    }
    for c in SLOT_COLS:
        v = g(c).str.strip().to_numpy(dtype=object)
        codes, uni = pd.factorize(v)
        out[c] = np.array([_slot_norm(x) for x in uni], dtype=object)[codes] if len(uni) else v
    return pd.DataFrame(out)


_REC_EMPTY = ["id", "name", "trl", "core", "addr", "pin", "miss", "nonascii", "web", "src",
              "prim", "alias", "street", "landmark"] + SLOT_COLS


def _load_records(files, country):
    parts = []
    for src, f in enumerate(files):
        path = ART / f
        header = pd.read_csv(path, nrows=0).columns
        use = [c for c in REC_COLS if c in header]
        for ch in pd.read_csv(path, usecols=use, dtype=str, keep_default_na=False,
                              chunksize=READ_CHUNK):
            ch = ch[ch["country_clean"].str.strip() == country]
            if len(ch):
                parts.append(_reduce(ch, src))
    if not parts:
        return pd.DataFrame(columns=_REC_EMPTY)
    df = pd.concat(parts, ignore_index=True)
    return df.drop_duplicates("id", keep="first").reset_index(drop=True)


# ============================================================================
#  per-country vector space (built once, on unique strings)
# ============================================================================
def _binary(docs, pattern=r"(?u)\S+", vocab=False):
    try:
        cv = CountVectorizer(token_pattern=pattern, lowercase=False,
                             binary=True, dtype=np.float32)
        M = cv.fit_transform(docs).tocsr()
        V = np.asarray(cv.get_feature_names_out(), dtype=object)
    except ValueError:                                   # empty vocabulary
        M = sparse.csr_matrix((len(docs), 1), dtype=np.float32)
        V = np.array([""], dtype=object)
    return (M, V) if vocab else M


def _char(docs):
    try:
        cv = CountVectorizer(analyzer="char_wb", ngram_range=CHAR_NGRAM,
                             lowercase=False, dtype=np.float32)
        return cv.fit_transform(docs).tocsr()
    except ValueError:
        return sparse.csr_matrix((len(docs), 1), dtype=np.float32)


def _idf(M, cnt, n_rec):
    """Smoothed IDF from RECORD frequency (unique strings weighted by how many
    records carry them), divided by its max so it is 0..1 in every country."""
    B = M.copy()
    B.data[:] = 1.0
    df = np.asarray(B.T @ cnt).ravel()
    idf = np.log((1.0 + n_rec) / (1.0 + df)) + 1.0
    return (idf / (np.log(1.0 + n_rec) + 1.0)).astype(np.float32)


def _scale_cols(M, w):
    return (M @ sparse.diags(w)).tocsr().astype(np.float32)


def _topk(T, w, K, colmask=None):
    """per row, its K highest-weight columns (rarest words) as a CSR token list:
    (indptr int64, indices int32, per-row L2 norm of their weights)."""
    T = T.tocsr()
    nr = T.shape[0]
    rows = np.repeat(np.arange(nr, dtype=np.int64), np.diff(T.indptr))
    cols = T.indices.astype(np.int64)
    if colmask is not None:
        k = colmask[cols]
        rows, cols = rows[k], cols[k]
    if len(cols):
        o = np.lexsort((-w[cols], rows))
        rows, cols = rows[o], cols[o]
        cnt = np.bincount(rows, minlength=nr)
        rank = np.arange(len(rows)) - (np.cumsum(cnt) - cnt)[rows]
        k = rank < K
        rows, cols = rows[k], cols[k]
    ptr = np.zeros(nr + 1, np.int64)
    ptr[1:] = np.cumsum(np.bincount(rows, minlength=nr))
    nrm = np.sqrt(np.bincount(rows, weights=w[cols].astype(np.float64) ** 2, minlength=nr))
    return ptr, cols.astype(np.int32), nrm.astype(np.float32)


def _build_space(R, n_s1, log):
    t = time.time()
    n_rec = len(R)
    S = {"n_s1": n_s1, "miss_rec": None}

    # ---------------- names ----------------
    key = np.where(R["web"].to_numpy(), (R["name"] + "\x00").to_numpy(), R["name"].to_numpy())
    nm_rec = _derive(key.astype(object), _norm_name)
    nm_code, nm_uni = pd.factorize(nm_rec)
    nm_uni = np.asarray(nm_uni, dtype=object)
    cnt = np.bincount(nm_code, minlength=len(nm_uni)).astype(np.float64)
    S["nm_code"], S["nm_uni"] = nm_code, nm_uni
    S["cmp_uni"] = np.array([x.replace(" ", "") for x in nm_uni], dtype=object)
    S["nm_len"] = np.fromiter((len(x) for x in S["cmp_uni"]), np.float32, len(nm_uni))
    S["nm_ft"] = _codes(np.array([x.split(" ", 1)[0] for x in nm_uni], dtype=object))

    T, V = _binary(nm_uni, vocab=True)
    idf = _idf(T, cnt, n_rec)
    S["T_nw"], S["W_nw"] = T, _scale_cols(T, idf)
    S["X_nw"] = normalize(S["W_nw"], norm="l2", copy=True)
    S["nnz_nw"] = np.diff(T.indptr).astype(np.float32)
    S["rar_nw"] = _ratio(np.asarray(S["W_nw"].sum(axis=1)).ravel().astype(np.float32), S["nnz_nw"])
    S["V_nw"], S["idf_nw"] = V, idf
    S["tk_nw"] = _topk(T, idf, K_NAME)

    # candidate-only noise words: far more frequent among candidate names than
    # among S1 names of the same country ('service', 'center', 'trading' ...)
    c_s1 = np.bincount(nm_code[:n_s1], minlength=len(nm_uni)).astype(np.float64)
    c_cd = cnt - c_s1
    Tb = T.copy()
    Tb.data[:] = 1.0
    df_s1 = np.asarray(Tb.T @ c_s1).ravel()
    df_cd = np.asarray(Tb.T @ c_cd).ravel()
    n_cd = max(n_rec - n_s1, 1)
    S["noise_nw"] = (df_cd >= NOISE_DF) & (df_cd / n_cd >= NOISE_RATIO * (df_s1 + 1.0) / max(n_s1, 1))
    del Tb
    nz = np.flatnonzero(S["noise_nw"])
    top_noise = [V[i] for i in nz[np.argsort(-df_cd[nz])][:12]]

    C = _char(nm_uni)
    idc = _idf(C, cnt, n_rec)
    C.data = 1.0 + np.log(C.data)                        # sublinear tf
    S["X_nc"] = normalize(_scale_cols(C, idc), norm="l2", copy=False)
    del C

    ph = np.array([_phon_name(x) for x in nm_uni], dtype=object)
    ph_code_of_nm, ph_uni = pd.factorize(ph)
    ph_uni = np.asarray(ph_uni, dtype=object)
    S["ph_of_nm"], S["ph_uni"] = ph_code_of_nm, ph_uni
    S["phc_uni"] = np.array([x.replace(" ", "") for x in ph_uni], dtype=object)
    Tp, Vp = _binary(ph_uni, vocab=True)
    S["T_pw"] = Tp
    S["nnz_pw"] = np.diff(Tp.indptr).astype(np.float32)
    cph = np.bincount(ph_code_of_nm[nm_code], minlength=len(ph_uni)).astype(np.float64)
    idp = _idf(Tp, cph, n_rec)
    S["V_pw"], S["idf_pw"] = Vp, idp
    S["tk_pw"] = _topk(Tp, idp, K_NAME)

    S["s1_name_cnt"] = np.bincount(nm_code[:n_s1], minlength=len(nm_uni)).astype(np.float32)

    # ---------------- legal form ----------------
    S["lf"] = _legal_masks(R["trl"].to_numpy(dtype=object), R["core"].to_numpy(dtype=object))

    # ---------------- addresses ----------------
    ad_rec = _derive(R["addr"].to_numpy(dtype=object), _norm_addr)
    ad_code, ad_uni = pd.factorize(ad_rec)
    ad_uni = np.asarray(ad_uni, dtype=object)
    acnt = np.bincount(ad_code, minlength=len(ad_uni)).astype(np.float64)
    S["ad_code"], S["ad_uni"] = ad_code, ad_uni
    S["ad_sorted"] = np.array([" ".join(sorted(set(x.split()))) for x in ad_uni], dtype=object)
    S["miss_rec"] = R["miss"].to_numpy(dtype=bool) | (ad_rec == "")

    T, Va = _binary(ad_uni, vocab=True)
    idf = _idf(T, acnt, n_rec)
    S["T_aw"], S["W_aw"] = T, _scale_cols(T, idf)
    S["X_aw"] = normalize(S["W_aw"], norm="l2", copy=True)
    S["nnz_aw"] = np.diff(T.indptr).astype(np.float32)
    alpha = np.array([len(x) >= 2 and x.isalpha() for x in Va], dtype=bool)
    S["V_aw"], S["idf_aw"] = Va, idf
    S["tk_aw"] = _topk(T, idf, K_ADDR, colmask=alpha)       # street / locality words

    Tn, Vn = _binary(ad_uni, pattern=r"(?<!\S)\d+(?!\S)", vocab=True)
    S["T_an"] = Tn
    S["nnz_an"] = np.diff(Tn.indptr).astype(np.float32)
    idn = _idf(Tn, acnt, n_rec)
    S["V_an"] = Vn
    S["val_an"] = np.array([float(x[-9:]) if x.isdigit() else np.nan for x in Vn], dtype=np.float64)
    S["tk_an"] = _topk(Tn, idn, K_NUM)
    fn = np.array([_first_num(x) for x in ad_uni], dtype=object)
    S["ad_fn"] = _codes(fn)
    S["ad_fn_str"] = fn
    S["ad_fn_val"] = np.array([float(x[-9:]) if x else np.nan for x in fn], dtype=np.float64)

    pins = R["pin"].to_numpy(dtype=object)
    S["pin"] = _codes(pins)
    S["pin3"] = _codes(np.array([p[:3] for p in pins], dtype=object))
    S["pin_str"] = pins
    S["nonascii"] = R["nonascii"].to_numpy(dtype=bool)
    S["web"] = R["web"].to_numpy(dtype=bool)

    # ================= final-version additions =================
    S["src"] = R["src"].to_numpy(dtype=np.int8)
    nu = len(nm_uni)
    toks_nm = [x.split() for x in nm_uni]
    # numbers written inside the name (local union 874, 21st century 2)
    S["nm_num"] = _codes(np.array([" ".join(sorted({t for t in tk if t.isdigit()}))
                                   for tk in toks_nm], dtype=object)).astype(np.int32)
    # initials of multi-word names (acronym matching)
    S["nm_init"] = np.array(["".join(t[0] for t in tk) if len(tk) >= 2 else "" for tk in toks_nm],
                            dtype=object)
    del toks_nm
    # how many candidate records carry the same name; how many S1 share a first word
    S["cd_name_cnt"] = np.bincount(nm_code[n_s1:], minlength=nu).astype(np.float32)
    ft_rec = S["nm_ft"][nm_code]
    nft = int(S["nm_ft"].max()) + 1 if nu else 1
    s1ft = ft_rec[:n_s1]
    S["s1_ft_cnt"] = np.bincount(s1ft[s1ft >= 0], minlength=max(nft, 1)).astype(np.float32)
    del ft_rec, s1ft
    S["W_nw_sum"] = np.asarray(S["W_nw"].sum(axis=1)).ravel().astype(np.float32)

    # how many S1 records sit at exactly this (non-missing) address
    ad_rec_code = np.where(S["miss_rec"], -1, ad_code)
    s1a = ad_rec_code[:n_s1]
    S["s1_addr_cnt"] = np.bincount(s1a[s1a >= 0], minlength=len(ad_uni)).astype(np.float32)
    del ad_rec_code, s1a

    # name words that appear in an address (same vocabulary as the name space)
    try:
        cvx = CountVectorizer(token_pattern=r"(?u)\S+", lowercase=False, binary=True,
                              dtype=np.float32, vocabulary={w: i for i, w in enumerate(S["V_nw"])
                                                            if w})
        A = cvx.transform(ad_uni).tocsr()
        if A.shape[1] < S["W_nw"].shape[1]:
            A = sparse.hstack([A, sparse.csr_matrix((A.shape[0], S["W_nw"].shape[1] - A.shape[1]),
                                                    dtype=np.float32)]).tocsr()
        S["A_nx"] = A
    except ValueError:
        S["A_nx"] = None

    # alias halves (DBA / formerly / aka) and the address fields from preprocessing
    for c, key in (("prim", "pr"), ("alias", "al"), ("street", "st"), ("landmark", "lm")):
        v = R[c].to_numpy(dtype=object)
        codes, uni = pd.factorize(v)
        codes = np.where(v == "", -1, codes).astype(np.int32)
        S[f"{key}_code"], S[f"{key}_uni"] = codes, np.asarray(uni, dtype=object)
    for c in SLOT_COLS:
        S[f"sl_{c}"] = _codes(R[c].to_numpy(dtype=object)).astype(np.int32)

    log(f"    vector space: {len(nm_uni):,} unique names, {len(ph_uni):,} phonetic keys, "
        f"{len(ad_uni):,} unique addresses | vocab name={S['T_nw'].shape[1]:,} "
        f"char={S['X_nc'].shape[1]:,} addr={S['T_aw'].shape[1]:,}   [{time.time() - t:.1f}s]")
    log(f"    candidate-only noise words: {int(S['noise_nw'].sum()):,}"
        + (f"  (most frequent: {', '.join(top_noise)})" if top_noise else "")
        + f";  legal form known for {float((S['lf'][:n_s1] != 0).mean()):.1%} of S1 / "
          f"{float((S['lf'][n_s1:] != 0).mean()) if n_rec > n_s1 else 0.0:.1%} of candidates")
    return S


# ============================================================================
#  vectorised row-wise sparse helpers
# ============================================================================
def _rowdot(A, ia, ib, B=None):
    B = A if B is None else B
    return np.asarray(A[ia].multiply(B[ib]).sum(axis=1), dtype=np.float32).ravel()


def _rowmax(M):
    return np.asarray(M.max(axis=1).todense(), dtype=np.float32).ravel()


def _shared_max(W, T, ia, ib):
    return _rowmax(W[ia].multiply(T[ib]).tocsr())


def _unshared_max(W, T, ia, ib):
    """max IDF over tokens of side A that side B does NOT have."""
    Wa = W[ia]
    return _rowmax((Wa - Wa.multiply(T[ib])).tocsr())


def _ratio(num, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / den
    return np.where(den > 0, r, np.nan).astype(np.float32)


def _sim(a, b, scorer, scale=1.0):
    r = cpdist(a.tolist(), b.tolist(), scorer=scorer, workers=WORKERS, dtype=np.float32)
    return (np.asarray(r, dtype=np.float32) / scale) if scale != 1.0 else np.asarray(r, dtype=np.float32)


def _eq(a, b):
    return np.where((a >= 0) & (b >= 0), (a == b).astype(np.float32), np.nan).astype(np.float32)


# ============================================================================
#  soft token matching (word x word cross product, flat numpy)
# ============================================================================
def _chunks_by_rows(tot, cap):
    """split pair indices so that each chunk has <= cap cross-product rows."""
    if len(tot) == 0:
        return []
    cs = np.cumsum(tot)
    out, lo = [], 0
    while lo < len(tot):
        base = cs[lo - 1] if lo else 0
        hi = int(np.searchsorted(cs, base + cap, side="right"))
        hi = max(hi, lo + 1)
        out.append((lo, min(hi, len(tot))))
        lo = hi
    return out


def _xprod(ptr, idx, a, b):
    """word x word rows of pairs (a[k], b[k]), ordered pair -> word of a -> word of b."""
    ca = (ptr[a + 1] - ptr[a]).astype(np.int64)
    cb = (ptr[b + 1] - ptr[b]).astype(np.int64)
    tot = ca * cb
    off = np.cumsum(tot) - tot
    R = int(tot.sum())
    pair = np.repeat(np.arange(len(a), dtype=np.int64), tot)
    m = np.arange(R, dtype=np.int64) - off[pair]
    cbr = cb[pair]
    i = m // cbr
    j = m - i * cbr
    ta = idx[ptr[a][pair] + i].astype(np.int64)
    tb = idx[ptr[b][pair] + j].astype(np.int64)
    return ca, cb, off, pair, m, i, j, ta, tb


def _soft(tk, vocab, idf, a, b, noise=None):
    """Soft-TFIDF, Monge-Elkan (both directions), IDF-weighted soft coverage of each
    side, and the words left unmatched — for pairs (a[k], b[k]) of unique strings.
    Two words match when Jaro-Winkler >= SOFT_THETA (exact words: 1.0)."""
    ptr, idx, nrm = tk
    n = len(a)
    keys = ["soft", "me_a", "me_b", "cov_a", "cov_b", "unm_a", "unm_b", "unm_b_noise",
            "unm_idf_a", "unm_idf_b"]
    out = {k: np.full(n, np.nan, np.float32) for k in keys}
    ca = ptr[a + 1] - ptr[a]
    cb = ptr[b + 1] - ptr[b]
    ok = np.flatnonzero((ca > 0) & (cb > 0))
    if not len(ok):
        return out
    V = np.int64(len(vocab))
    for lo, hi in _chunks_by_rows((ca[ok] * cb[ok]).astype(np.int64), ROW_CAP):
        k = ok[lo:hi]
        A, Bv = a[k], b[k]
        cA, cB, off, pair, m, i, j, ta, tb = _xprod(ptr, idx, A, Bv)
        eq = ta == tb
        np_ = len(k)
        # A-groups (pair, word of A): contiguous rows, size cB
        gsA = np.cumsum(cA) - cA
        pairA = np.repeat(np.arange(np_, dtype=np.int64), cA)
        iA = np.arange(int(cA.sum()), dtype=np.int64) - gsA[pairA]
        startA = off[pairA] + iA * cB[pairA]
        # B-groups (pair, word of B): same rows re-ordered word-of-B major, size cA
        gsB = np.cumsum(cB) - cB
        pairB = np.repeat(np.arange(np_, dtype=np.int64), cB)
        jB = np.arange(int(cB.sum()), dtype=np.int64) - gsB[pairB]
        cAr = cA[pair]
        jj = m // cAr
        permB = off[pair] + (m - jj * cAr) * cB[pair] + jj
        startB = off[pairB] + jB * cA[pairB]
        exA = np.logical_or.reduceat(eq, startA)
        exB = np.logical_or.reduceat(eq[permB], startB)
        # Jaro-Winkler only where a word has no exact partner, once per unique word pair
        need = ~eq & (~exA[gsA[pair] + i] | ~exB[gsB[pair] + j])
        s = eq.astype(np.float32)
        if need.any():
            kk = ta[need] * V + tb[need]
            uk, inv = np.unique(kk, return_inverse=True)
            jw = np.asarray(cpdist(vocab[uk // V].tolist(), vocab[uk % V].tolist(),
                                   scorer=JaroWinkler.normalized_similarity,
                                   workers=WORKERS, dtype=np.float32), dtype=np.float32)
            s[need] = jw[inv.ravel()]
        del need
        th = s >= SOFT_THETA
        # best partner of every word, both directions
        bestA = np.maximum.reduceat(s, startA)
        sP = s[permB]
        bestB = np.maximum.reduceat(sP, startB)
        tokA = ta[startA]
        tokB = tb[permB][startB]
        wA, wB = idf[tokA].astype(np.float32), idf[tokB].astype(np.float32)
        mA, mB = bestA >= SOFT_THETA, bestB >= SOFT_THETA
        # Soft-TFIDF: sum over words of V_a(w) * max over close partners of V_b(w') * sim
        nA = nrm[A][pair]
        nB = nrm[Bv][pair]
        vA_row = np.where(nA > 0, idf[ta] / np.maximum(nA, 1e-9), 0).astype(np.float32)
        vB_row = np.where(nB > 0, idf[tb] / np.maximum(nB, 1e-9), 0).astype(np.float32)
        tA = np.where(th, vB_row * s, 0).astype(np.float32)
        tB = np.where(th, vA_row * s, 0).astype(np.float32)[permB]
        del th, vA_row, vB_row, sP
        xA = np.maximum.reduceat(tA, startA)
        xB = np.maximum.reduceat(tB, startB)
        vA = wA / np.maximum(nrm[A][pairA], 1e-9)
        vB = wB / np.maximum(nrm[Bv][pairB], 1e-9)
        sa = np.add.reduceat(vA * xA, gsA)
        sb = np.add.reduceat(vB * xB, gsB)
        out["soft"][k] = np.minimum((sa + sb) / 2.0, 1.0)
        out["me_a"][k] = np.add.reduceat(bestA, gsA) / cA
        out["me_b"][k] = np.add.reduceat(bestB, gsB) / cB
        out["cov_a"][k] = _ratio(np.add.reduceat(wA * bestA * mA, gsA), np.add.reduceat(wA, gsA))
        out["cov_b"][k] = _ratio(np.add.reduceat(wB * bestB * mB, gsB), np.add.reduceat(wB, gsB))
        out["unm_a"][k] = np.add.reduceat((~mA).astype(np.float32), gsA)
        out["unm_idf_a"][k] = np.maximum.reduceat(np.where(mA, 0, wA).astype(np.float32), gsA)
        if noise is not None:
            nzB = noise[tokB]
            real = ~mB & ~nzB
            out["unm_b"][k] = np.add.reduceat(real.astype(np.float32), gsB)
            out["unm_b_noise"][k] = np.add.reduceat((~mB & nzB).astype(np.float32), gsB)
            out["unm_idf_b"][k] = np.maximum.reduceat(np.where(real, wB, 0).astype(np.float32), gsB)
        else:
            out["unm_b"][k] = np.add.reduceat((~mB).astype(np.float32), gsB)
            out["unm_idf_b"][k] = np.maximum.reduceat(np.where(mB, 0, wB).astype(np.float32), gsB)
        del cA, cB, off, pair, m, i, j, ta, tb, eq, s, permB
    return out


def _num_close(tk, vocab, val, a, b):
    """best agreement between the house numbers of two addresses:
    Levenshtein similarity of the digit strings and numeric closeness
    1 - |x - y| / max(x, y) — both 1.0 when a number is shared exactly."""
    ptr, idx, _ = tk
    n = len(a)
    lev = np.full(n, np.nan, np.float32)
    clo = np.full(n, np.nan, np.float32)
    ca = ptr[a + 1] - ptr[a]
    cb = ptr[b + 1] - ptr[b]
    ok = np.flatnonzero((ca > 0) & (cb > 0))
    if not len(ok):
        return lev, clo
    V = np.int64(len(vocab))
    for lo, hi in _chunks_by_rows((ca[ok] * cb[ok]).astype(np.int64), ROW_CAP):
        k = ok[lo:hi]
        cA, cB, off, pair, m, i, j, ta, tb = _xprod(ptr, idx, a[k], b[k])
        eq = ta == tb
        lv = np.ones(len(ta), np.float32)
        ne = ~eq
        if ne.any():
            kk = ta[ne] * V + tb[ne]
            uk, inv = np.unique(kk, return_inverse=True)
            d = np.asarray(cpdist(vocab[uk // V].tolist(), vocab[uk % V].tolist(),
                                  scorer=Levenshtein.normalized_similarity,
                                  workers=WORKERS, dtype=np.float32), dtype=np.float32)
            lv[ne] = d[inv.ravel()]
        x, y = val[ta], val[tb]
        with np.errstate(divide="ignore", invalid="ignore"):
            c = 1.0 - np.abs(x - y) / np.maximum(np.maximum(x, y), 1.0)
        c = np.where(np.isfinite(c), c, 0.0).astype(np.float32)
        c[eq] = 1.0
        lev[k] = np.maximum.reduceat(lv, off)
        clo[k] = np.maximum.reduceat(c, off)
    return lev, clo


# ============================================================================
#  final-version helpers
# ============================================================================
def _is_abbr(short, init):
    """short = compact one-word name, init = initials of a multi-word name.
    lhs ~ lucky hair studio, sb ~ shivam business, dicarecom ~ dental interstate care ..."""
    if len(init) < 2 or not short or len(short) > ABBR_MAX + 3:
        return False
    return short == init or short.startswith(init)


def _abbrev(S, n1, n2, ok_n):
    out = np.where(ok_n, 0.0, np.nan).astype(np.float32)
    t1, t2 = S["nnz_nw"][n1], S["nnz_nw"][n2]
    cand = np.flatnonzero(ok_n & (((t1 == 1) & (t2 >= 2)) | ((t2 == 1) & (t1 >= 2))))
    if not len(cand):
        return out
    M = np.int64(len(S["nm_uni"]))
    uk, inv = np.unique(n1[cand].astype(np.int64) * M + n2[cand], return_inverse=True)
    a, b = uk // M, uk % M
    cmp_, ini = S["cmp_uni"], S["nm_init"]
    res = np.fromiter((_is_abbr(cmp_[x], ini[y]) or _is_abbr(cmp_[y], ini[x])
                       for x, y in zip(a, b)), np.float32, len(uk))
    out[cand] = res[inv.ravel()]
    return out


def _uniq_sim(u1, u2, c1, c2, scorer, scale=1.0):
    """similarity of strings u1[c1] vs u2[c2], computed once per unique code pair;
    NaN where either code is -1 (missing)."""
    out = np.full(len(c1), np.nan, np.float32)
    ok = np.flatnonzero((c1 >= 0) & (c2 >= 0))
    if not len(ok):
        return out
    M = np.int64(max(len(u2), 1))
    uk, inv = np.unique(c1[ok].astype(np.int64) * M + c2[ok], return_inverse=True)
    r = _sim(u1[uk // M], u2[uk % M], scorer, scale)
    out[ok] = r[inv.ravel()]
    return out


def _alias_feats(S, n1, n2, i1, i2, F):
    pr1, al1 = S["pr_code"][i1], S["al_code"][i1]
    pr2, al2 = S["pr_code"][i2], S["al_code"][i2]
    has = (al1 >= 0) | (al2 >= 0)
    F[:, FIDX["alias_any"]] = has.astype(np.float32)
    k = np.flatnonzero(has)
    if not len(k):
        return
    E = np.array([""], dtype=object)

    def pick(uni, code):
        return np.where(code >= 0, np.concatenate([uni, E])[np.where(code >= 0, code, len(uni))], "")

    v1 = [S["nm_uni"][n1[k]], pick(S["pr_uni"], pr1[k]), pick(S["al_uni"], al1[k])]
    v2 = [S["nm_uni"][n2[k]], pick(S["pr_uni"], pr2[k]), pick(S["al_uni"], al2[k])]
    bt = np.full(len(k), -1.0, np.float32)
    bj = np.full(len(k), -1.0, np.float32)
    ex = np.zeros(len(k), bool)
    for A in v1:
        for B in v2:
            ok = np.flatnonzero((A != "") & (B != ""))
            if not len(ok):
                continue
            a, b = A[ok], B[ok]
            bt[ok] = np.maximum(bt[ok], _sim(a, b, fuzz.token_set_ratio, 100.0))
            bj[ok] = np.maximum(bj[ok], _sim(a, b, JaroWinkler.normalized_similarity))
            ex[ok] |= (a == b)
    bt[bt < 0] = np.nan
    bj[bj < 0] = np.nan
    F[k, FIDX["alias_best_tset"]] = bt
    F[k, FIDX["alias_best_jw"]] = bj
    F[k, FIDX["alias_exact"]] = ex.astype(np.float32)
    F[k, FIDX["alias_gain"]] = bt - F[k, FIDX["name_token_set"]]


def _field_feats(S, i1, i2, a1, a2, ok_a, F):
    F[:, FIDX["addr_exact_eq"]] = np.where(ok_a, (a1 == a2).astype(np.float32), np.nan)
    F[:, FIDX["addr_idf_shared_sum"]] = np.where(ok_a, _rowdot(S["W_aw"], a1, a2, S["T_aw"]), np.nan)
    n = len(i1)
    nb = np.zeros(n, np.float32)
    ne = np.zeros(n, np.float32)
    sb = np.zeros(n, np.float32)
    se = np.zeros(n, np.float32)
    for c in SLOT_COLS:
        x, y = S[f"sl_{c}"][i1], S[f"sl_{c}"][i2]
        b = (x >= 0) & (y >= 0)
        e = b & (x == y)
        nb += b
        ne += e
        if c == "house_number":
            F[:, FIDX["f_house_eq"]] = np.where(b, e.astype(np.float32), np.nan)
        elif c == "unit_number":
            F[:, FIDX["f_unit_eq"]] = np.where(b, e.astype(np.float32), np.nan)
        if c in SUBLOC_COLS:
            sb += b
            se += e
    F[:, FIDX["f_subloc_eq"]] = _ratio(se, sb)
    F[:, FIDX["fld_n_both"]] = nb
    F[:, FIDX["fld_n_eq"]] = ne
    F[:, FIDX["fld_n_conflict"]] = nb - ne
    st1, st2 = S["st_code"][i1], S["st_code"][i2]
    F[:, FIDX["f_street_tset"]] = _uniq_sim(S["st_uni"], S["st_uni"], st1, st2, fuzz.token_set_ratio, 100.0)
    F[:, FIDX["f_street_jw"]] = _uniq_sim(S["st_uni"], S["st_uni"], st1, st2,
                                          JaroWinkler.normalized_similarity)
    F[:, FIDX["f_landmark_tset"]] = _uniq_sim(S["lm_uni"], S["lm_uni"], S["lm_code"][i1],
                                              S["lm_code"][i2], fuzz.token_set_ratio, 100.0)


# ============================================================================
#  features for one batch of pairs
# ============================================================================
def _batch_features(S, i1, i2, F):
    """i1: S1 record rows, i2: candidate record rows (both into R). Fills F (n x FEATURES)."""
    # ---------------- names ----------------
    n1, n2 = S["nm_code"][i1], S["nm_code"][i2]
    s1n, s2n = S["nm_uni"][n1], S["nm_uni"][n2]
    c1, c2 = S["cmp_uni"][n1], S["cmp_uni"][n2]
    ok_n = (S["nm_len"][n1] > 0) & (S["nm_len"][n2] > 0)

    F[:, FIDX["name_lev"]] = _sim(s1n, s2n, Levenshtein.normalized_similarity)
    F[:, FIDX["name_jw"]] = _sim(s1n, s2n, JaroWinkler.normalized_similarity)
    F[:, FIDX["name_token_sort"]] = _sim(s1n, s2n, fuzz.token_sort_ratio, 100.0)
    F[:, FIDX["name_token_set"]] = _sim(s1n, s2n, fuzz.token_set_ratio, 100.0)
    F[:, FIDX["name_partial_compact"]] = _sim(c1, c2, fuzz.partial_ratio, 100.0)
    F[:, FIDX["name_compact_lev"]] = _sim(c1, c2, Levenshtein.normalized_similarity)
    F[:, FIDX["name_char_tfidf_cos"]] = _rowdot(S["X_nc"], n1, n2)
    F[:, FIDX["name_word_tfidf_cos"]] = _rowdot(S["X_nw"], n1, n2)

    inter = _rowdot(S["T_nw"], n1, n2)
    a, b = S["nnz_nw"][n1], S["nnz_nw"][n2]
    F[:, FIDX["name_jaccard"]] = _ratio(inter, a + b - inter)
    F[:, FIDX["name_overlap"]] = _ratio(inter, np.minimum(a, b))
    F[:, FIDX["name_idf_shared_max"]] = _shared_max(S["W_nw"], S["T_nw"], n1, n2)
    F[:, FIDX["name_idf_unshared_s1"]] = _unshared_max(S["W_nw"], S["T_nw"], n1, n2)
    F[:, FIDX["name_idf_unshared_cand"]] = _unshared_max(S["W_nw"], S["T_nw"], n2, n1)
    F[:, FIDX["name_set_eq"]] = ((inter == a) & (a == b)).astype(np.float32)
    F[:, FIDX["name_ntok_s1"]] = a
    F[:, FIDX["name_ntok_cand"]] = b
    F[:, FIDX["name_rarity_s1"]] = S["rar_nw"][n1]

    p1, p2 = S["ph_of_nm"][n1], S["ph_of_nm"][n2]
    F[:, FIDX["name_phon_lev"]] = _sim(S["phc_uni"][p1], S["phc_uni"][p2],
                                       Levenshtein.normalized_similarity)
    pint = _rowdot(S["T_pw"], p1, p2)
    pa_, pb_ = S["nnz_pw"][p1], S["nnz_pw"][p2]
    F[:, FIDX["name_phon_jaccard"]] = _ratio(pint, pa_ + pb_ - pint)
    F[:, FIDX["name_first_tok_eq"]] = _eq(S["nm_ft"][n1], S["nm_ft"][n2])
    l1, l2 = S["nm_len"][n1], S["nm_len"][n2]
    F[:, FIDX["name_len_ratio"]] = _ratio(np.minimum(l1, l2), np.maximum(l1, l2))

    # soft word matching on the name (words matching up to a typo / transliteration)
    so = _soft(S["tk_nw"], S["V_nw"], S["idf_nw"], n1, n2, noise=S["noise_nw"])
    F[:, FIDX["name_soft_tfidf"]] = so["soft"]
    F[:, FIDX["name_me_s1"]] = so["me_a"]
    F[:, FIDX["name_me_cand"]] = so["me_b"]
    F[:, FIDX["name_softcov_s1"]] = so["cov_a"]
    F[:, FIDX["name_softcov_cand"]] = so["cov_b"]
    F[:, FIDX["name_unm_s1"]] = so["unm_a"]
    F[:, FIDX["name_unm_cand_real"]] = so["unm_b"]
    F[:, FIDX["name_unm_cand_noise"]] = so["unm_b_noise"]
    F[:, FIDX["name_unm_idf_s1"]] = so["unm_idf_a"]
    F[:, FIDX["name_unm_idf_cand"]] = so["unm_idf_b"]
    # every S1 word found, and the candidate's only extra words are source noise
    F[:, FIDX["name_denoised_eq"]] = np.where(np.isnan(so["unm_a"]), np.nan,
                                              ((so["unm_a"] == 0) & (so["unm_b"] == 0))
                                              .astype(np.float32))
    del so

    sp = _soft(S["tk_pw"], S["V_pw"], S["idf_pw"], p1, p2)
    F[:, FIDX["phon_soft_tfidf"]] = sp["soft"]
    F[:, FIDX["phon_softcov_s1"]] = sp["cov_a"]
    F[:, FIDX["phon_softcov_cand"]] = sp["cov_b"]
    del sp

    F[np.ix_(~ok_n, [FIDX[f] for f in _NAME_SIM])] = np.nan
    # phonetic key can be empty for non-Latin leftovers even when the name is not
    ph_empty = (S["phc_uni"][p1] == "") | (S["phc_uni"][p2] == "")
    for f in ("name_phon_lev", "name_phon_jaccard", "phon_soft_tfidf",
              "phon_softcov_s1", "phon_softcov_cand"):
        F[ph_empty, FIDX[f]] = np.nan

    F[:, FIDX["name_script_mismatch"]] = (S["nonascii"][i1] != S["nonascii"][i2]).astype(np.float32)
    F[:, FIDX["name_is_web"]] = (S["web"][i1] | S["web"][i2]).astype(np.float32)

    # ---------------- legal form ----------------
    la, lb = S["lf"][i1], S["lf"][i2]
    ha, hb = la != 0, lb != 0
    both = ha & hb
    F[:, FIDX["lf_s1"]] = ha.astype(np.float32)
    F[:, FIDX["lf_cand"]] = hb.astype(np.float32)
    F[:, FIDX["lf_eq"]] = np.where(both, (la == lb).astype(np.float32), np.nan)
    F[:, FIDX["lf_jacc"]] = np.where(both, _ratio(_popcount(la & lb), _popcount(la | lb)), np.nan)

    # ---------------- addresses ----------------
    m1, m2 = S["miss_rec"][i1], S["miss_rec"][i2]
    F[:, FIDX["addr_missing_s1"]] = m1.astype(np.float32)
    F[:, FIDX["addr_missing_cand"]] = m2.astype(np.float32)
    ok_a = ~(m1 | m2)

    a1, a2 = S["ad_code"][i1], S["ad_code"][i2]
    F[:, FIDX["addr_token_set"]] = _sim(S["ad_uni"][a1], S["ad_uni"][a2], fuzz.token_set_ratio, 100.0)
    F[:, FIDX["addr_sorted_lev"]] = _sim(S["ad_sorted"][a1], S["ad_sorted"][a2],
                                         Levenshtein.normalized_similarity)
    F[:, FIDX["addr_word_tfidf_cos"]] = _rowdot(S["X_aw"], a1, a2)
    inter = _rowdot(S["T_aw"], a1, a2)
    a, b = S["nnz_aw"][a1], S["nnz_aw"][a2]
    F[:, FIDX["addr_jaccard"]] = _ratio(inter, a + b - inter)
    F[:, FIDX["addr_overlap"]] = _ratio(inter, np.minimum(a, b))
    F[:, FIDX["addr_idf_shared_max"]] = _shared_max(S["W_aw"], S["T_aw"], a1, a2)
    F[:, FIDX["addr_idf_unshared_cand"]] = _unshared_max(S["W_aw"], S["T_aw"], a2, a1)
    F[:, FIDX["addr_len_ratio"]] = _ratio(np.minimum(a, b), np.maximum(a, b))

    ninter = _rowdot(S["T_an"], a1, a2)
    na, nb = S["nnz_an"][a1], S["nnz_an"][a2]
    has_num = (na > 0) & (nb > 0)
    F[:, FIDX["addr_num_jaccard"]] = np.where(has_num, _ratio(ninter, na + nb - ninter), np.nan)
    F[:, FIDX["addr_num_overlap"]] = np.where(has_num, _ratio(ninter, np.minimum(na, nb)), np.nan)
    F[:, FIDX["addr_first_num_eq"]] = _eq(S["ad_fn"][a1], S["ad_fn"][a2])

    # soft street / locality words (typos: stephen ~ stsphen, elverta ~ elverrta)
    sa = _soft(S["tk_aw"], S["V_aw"], S["idf_aw"], a1, a2)
    F[:, FIDX["addr_soft_tfidf"]] = sa["soft"]
    F[:, FIDX["addr_softcov_s1"]] = sa["cov_a"]
    F[:, FIDX["addr_softcov_cand"]] = sa["cov_b"]
    F[:, FIDX["addr_me_cand"]] = sa["me_b"]
    F[:, FIDX["addr_unm_idf_cand"]] = sa["unm_idf_b"]
    del sa
    # house numbers: near misses (4460 ~ 4469, 460 ~ 4460, 7701 ~ 7712)
    lev, clo = _num_close(S["tk_an"], S["V_an"], S["val_an"], a1, a2)
    F[:, FIDX["addr_num_best_lev"]] = lev
    F[:, FIDX["addr_num_best_close"]] = clo
    # the house number (first number of the address): 4460 ~ 460, 2146 ~ 2148
    f1, f2 = S["ad_fn"][a1], S["ad_fn"][a2]
    hf = (f1 >= 0) & (f2 >= 0)
    hl = np.full(len(i1), np.nan, np.float32)
    hc = np.full(len(i1), np.nan, np.float32)
    hl[hf] = 1.0
    hc[hf] = 1.0
    df_ = np.flatnonzero(hf & (f1 != f2))
    if len(df_):
        hl[df_] = _sim(S["ad_fn_str"][a1[df_]], S["ad_fn_str"][a2[df_]], Levenshtein.normalized_similarity)
        x, y = S["ad_fn_val"][a1[df_]], S["ad_fn_val"][a2[df_]]
        hc[df_] = (1.0 - np.abs(x - y) / np.maximum(np.maximum(x, y), 1.0)).astype(np.float32)
    F[:, FIDX["addr_hnum_lev"]] = hl
    F[:, FIDX["addr_hnum_close"]] = hc
    street = F[:, FIDX["addr_softcov_cand"]]
    F[:, FIDX["addr_street_same_num_diff"]] = np.where(
        hf & ~np.isnan(street), ((street >= 0.8) & (f1 != f2)).astype(np.float32), np.nan)

    F[np.ix_(~ok_a, [FIDX[f] for f in _ADDR_SIM])] = np.nan

    # postal code is its own field — usable even when the street text is thin
    F[:, FIDX["pin_eq"]] = _eq(S["pin"][i1], S["pin"][i2])
    F[:, FIDX["pin_prefix_eq"]] = _eq(S["pin3"][i1], S["pin3"][i2])
    pv = np.full(len(i1), np.nan, np.float32)
    hp = np.flatnonzero((S["pin"][i1] >= 0) & (S["pin"][i2] >= 0))
    if len(hp):
        pv[hp] = _sim(S["pin_str"][i1[hp]], S["pin_str"][i2[hp]], Levenshtein.normalized_similarity)
    F[:, FIDX["pin_lev"]] = pv

    # ---------------- NAME+ (final) ----------------
    F[:, FIDX["name_idf_shared_sum"]] = np.where(ok_n, _rowdot(S["W_nw"], n1, n2, S["T_nw"]), np.nan)
    F[:, FIDX["name_rarity_cand"]] = S["rar_nw"][n2]
    u1, u2 = S["nm_num"][n1], S["nm_num"][n2]
    F[:, FIDX["name_num_eq"]] = _eq(u1, u2)
    F[:, FIDX["name_num_one_side"]] = np.where(ok_n, ((u1 >= 0) != (u2 >= 0)).astype(np.float32), np.nan)
    F[:, FIDX["name_abbrev"]] = _abbrev(S, n1, n2, ok_n)

    # ---------------- ALIAS / DBA halves ----------------
    _alias_feats(S, n1, n2, i1, i2, F)

    # ---------------- address fields ----------------
    _field_feats(S, i1, i2, a1, a2, ok_a, F)

    # ---------------- name words inside the other record's address ----------------
    if S["A_nx"] is not None:
        d1, d2 = S["W_nw_sum"][n1], S["W_nw_sum"][n2]
        x1 = _rowdot(S["W_nw"], n1, a2, S["A_nx"])
        x2 = _rowdot(S["W_nw"], n2, a1, S["A_nx"])
        F[:, FIDX["x_s1name_in_candaddr"]] = np.where(ok_n & ~m2 & (d1 > 0), x1 / np.maximum(d1, 1e-9), np.nan)
        F[:, FIDX["x_candname_in_s1addr"]] = np.where(ok_n & ~m1 & (d2 > 0), x2 / np.maximum(d2, 1e-9), np.nan)

    # ---------------- no candidate address: the name must carry the decision ----------------
    F[:, FIDX["noaddr_name_soft"]] = np.where(m2, F[:, FIDX["name_soft_tfidf"]], np.nan)
    F[:, FIDX["noaddr_name_eq"]] = np.where(m2 & ok_n, (n1 == n2).astype(np.float32), np.nan)


# ============================================================================
#  context features (need the whole country's pairs at once)
# ============================================================================
def _gap(v, g):
    mx = pd.Series(v).groupby(g).transform("max").to_numpy(dtype=np.float32)
    return (v - mx).astype(np.float32)


def _rank_other(g, v):
    """rank of each pair inside its group (0 = best, by descending v) and
    v minus the best OTHER value of the group (> 0 only for a clear leader).
    NaN values rank last and get NaN; a group of one gets NaN for the lead."""
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
    other_o = np.where(np.arange(n) == first, sec, best)
    other = np.empty(n, np.float64)
    other[o] = other_o
    with np.errstate(invalid="ignore"):
        lead = (vf - other).astype(np.float32)
    bad = np.isnan(v) | ~np.isfinite(other)
    lead[bad] = np.nan
    rank[np.isnan(v)] = np.nan
    return rank, lead


def _context_features(S, i1, i2, F):
    F[:, FIDX["s1_n_cand"]] = np.bincount(i1)[i1].astype(np.float32)
    F[:, FIDX["cand_n_s1"]] = np.bincount(i2)[i2].astype(np.float32)
    nv = F[:, FIDX["name_char_tfidf_cos"]]
    av = F[:, FIDX["addr_word_tfidf_cos"]]
    F[:, FIDX["name_gap_cand"]] = _gap(nv, i2)
    F[:, FIDX["addr_gap_cand"]] = _gap(av, i2)
    F[:, FIDX["name_gap_s1"]] = _gap(nv, i1)
    F[:, FIDX["addr_gap_s1"]] = _gap(av, i1)
    n1, n2 = S["nm_code"][i1], S["nm_code"][i2]
    ok = (S["nm_len"][n1] > 0) & (S["nm_len"][n2] > 0)
    F[:, FIDX["s1_name_dup"]] = np.where(ok, S["s1_name_cnt"][n1], np.nan)
    F[:, FIDX["cand_name_ambig"]] = np.where(ok, S["s1_name_cnt"][n2], np.nan)

    # blocking probability: where this pair stands among its S1's and its candidate's options
    pb = F[:, FIDX["blk_p_block"]]
    if np.isfinite(pb).any():
        F[:, FIDX["pb_rank_s1"]], F[:, FIDX["pb_vs_other_s1"]] = _rank_other(i1, pb)
        _, F[:, FIDX["pb_vs_other_cand"]] = _rank_other(i2, pb)   # its rank = blk_cand_rank
        pz = np.nan_to_num(pb, nan=0.0).astype(np.float64)
        F[:, FIDX["s1_pb_sum"]] = np.bincount(i1, weights=pz)[i1].astype(np.float32)
        F[:, FIDX["s1_n_strong"]] = np.bincount(i1, weights=(pz >= STRONG_P))[i1].astype(np.float32)
        F[:, FIDX["cand_pb_sum"]] = np.bincount(i2, weights=pz)[i2].astype(np.float32)
    ns = F[:, FIDX["name_soft_tfidf"]]
    F[:, FIDX["nsoft_rank_s1"]], F[:, FIDX["nsoft_vs_other_s1"]] = _rank_other(i1, ns)
    _, F[:, FIDX["nsoft_vs_other_cand"]] = _rank_other(i2, ns)
    asf = F[:, FIDX["addr_soft_tfidf"]]
    _, F[:, FIDX["asoft_vs_other_s1"]] = _rank_other(i1, asf)
    _, F[:, FIDX["asoft_vs_other_cand"]] = _rank_other(i2, asf)

    # ================= final-version context =================
    src = S["src"][i2].astype(np.int64)
    F[:, FIDX["cand_src"]] = src.astype(np.float32)
    if np.isfinite(pb).any():
        rc, _ = _rank_other(i2, pb)
        rs = F[:, FIDX["pb_rank_s1"]]
        F[:, FIDX["mutual_best_pb"]] = np.where(np.isnan(pb), np.nan, ((rs == 0) & (rc == 0)).astype(np.float32))
        F[:, FIDX["pb_rank_s1_src"]], _ = _rank_other(i1.astype(np.int64) * (int(src.max()) + 1) + src, pb)

    # one combined name + address score (name alone when an address is missing)
    nsz = F[:, FIDX["name_soft_tfidf"]]
    combo = np.where(np.isnan(asf), nsz, np.where(np.isnan(nsz), asf, 0.5 * nsz + 0.5 * asf)).astype(np.float32)
    F[:, FIDX["combo"]] = combo
    F[:, FIDX["combo_rank_s1"]], F[:, FIDX["combo_vs_other_s1"]] = _rank_other(i1, combo)
    rcc, F[:, FIDX["combo_vs_other_cand"]] = _rank_other(i2, combo)
    F[:, FIDX["mutual_best_combo"]] = np.where(
        np.isnan(combo), np.nan, ((F[:, FIDX["combo_rank_s1"]] == 0) & (rcc == 0)).astype(np.float32))

    # live ambiguity: other candidates of this S1 with the same name as this candidate,
    # other S1 of this candidate with the same name as this S1
    NU = np.int64(max(len(S["nm_uni"]), 1))
    for key, f in ((i1.astype(np.int64) * NU + n2, "s1_n_cand_same_name"),
                   (i2.astype(np.int64) * NU + n1, "cand_n_s1_same_name")):
        _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        F[:, FIDX[f]] = np.where(ok, cnt[inv.ravel()] - 1, np.nan).astype(np.float32)

    ft2 = S["nm_ft"][n2]
    F[:, FIDX["cand_fw_s1df"]] = np.where(ok & (ft2 >= 0), S["s1_ft_cnt"][np.maximum(ft2, 0)], np.nan)
    F[:, FIDX["name_cd_cnt_cand"]] = np.where(ok, S["cd_name_cnt"][n2], np.nan)
    m1, m2 = S["miss_rec"][i1], S["miss_rec"][i2]
    F[:, FIDX["cand_addr_s1_cnt"]] = np.where(m2, np.nan, S["s1_addr_cnt"][S["ad_code"][i2]])
    F[:, FIDX["s1_addr_dup"]] = np.where(m1, np.nan, S["s1_addr_cnt"][S["ad_code"][i1]])

    # sibling: the S1's best OTHER candidate (by p_block, else by the combined score)
    v = pb if np.isfinite(pb).any() else combo
    anc = _best_other(i1, v)
    k = np.flatnonzero(anc >= 0)
    if len(k):
        a_ = anc[k]
        F[k, FIDX["sib_anchor_pb"]] = v[a_]
        c_, s_ = i2[k], i2[a_]
        F[k, FIDX["sib_name_tset"]] = _uniq_sim(S["nm_uni"], S["nm_uni"], S["nm_code"][c_],
                                                S["nm_code"][s_], fuzz.token_set_ratio, 100.0)
        ac = np.where(S["miss_rec"][c_], -1, S["ad_code"][c_])
        as_ = np.where(S["miss_rec"][s_], -1, S["ad_code"][s_])
        F[k, FIDX["sib_addr_tset"]] = _uniq_sim(S["ad_uni"], S["ad_uni"], ac, as_, fuzz.token_set_ratio, 100.0)
        F[k, FIDX["sib_pin_eq"]] = _eq(S["pin"][c_], S["pin"][s_])


def _best_other(g, v):
    """for each row, the row index of the best OTHER row of its group by v (NaN last);
    -1 for a group of one."""
    n = len(v)
    vf = np.where(np.isnan(v), -np.inf, v).astype(np.float64)
    o = np.lexsort((-vf, g))
    go = g[o]
    st = np.flatnonzero(np.concatenate(([True], go[1:] != go[:-1])))
    gl = np.diff(np.append(st, n))
    first = np.repeat(st, gl)
    best = o[first]
    sec = np.repeat(np.where(gl >= 2, o[np.minimum(st + 1, n - 1)], -1), gl)
    oth = np.where(np.arange(n) == first, sec, best)
    out = np.empty(n, np.int64)
    out[o] = oth
    return out


# ============================================================================
#  output
# ============================================================================
class _Writer:
    def __init__(self, path_base, log):
        self.log = log
        self.path = path_base.with_suffix(".parquet" if HAVE_ARROW else ".tsv")
        if self.path.exists():
            self.path.unlink()
        self.pq = None
        self.schema = None
        self.first = True

    def write(self, df):
        if HAVE_ARROW:
            if self.pq is None:
                tbl = pa.Table.from_pandas(df, preserve_index=False)
                self.schema = tbl.schema
                self.pq = pq.ParquetWriter(self.path, self.schema, compression="zstd")
            else:
                tbl = pa.Table.from_pandas(df, schema=self.schema, preserve_index=False)
            self.pq.write_table(tbl, row_group_size=1_000_000)
        else:
            df.to_csv(self.path, sep="\t", index=False, mode="w" if self.first else "a",
                      header=self.first, na_rep="", float_format="%.6g")
        self.first = False

    def close(self):
        if self.pq is not None:
            self.pq.close()
        if not HAVE_ARROW:
            self.log("  (pyarrow not installed -> wrote TSV; `pip install pyarrow` "
                     "for a ~5x smaller, ~10x faster parquet file)")


# ============================================================================
#  report (train only): missing rate, class means, univariate AUC, redundancy
# ============================================================================
def _auc_fast(y, v):
    """rank-based AUC (ties averaged) on non-missing rows."""
    ok = ~np.isnan(v)
    y, v = y[ok], v[ok]
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0 or len(v) < 100 or np.nanstd(v) == 0:
        return np.nan
    r = pd.Series(v).rank().to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _report(sample, log):
    y = sample["label"].to_numpy()
    if len(np.unique(y)) < 2:
        return
    log("\n" + "=" * 100)
    log(f"FEATURE REPORT  (random sample of {len(sample):,} pairs, positives {y.mean():.3f})")
    log("=" * 100)
    log(f"  {'feature':<28}{'missing':>9}{'mean|y=1':>11}{'mean|y=0':>11}{'AUC':>8}{'|AUC-.5|':>10}")
    log("  " + "-" * 75)
    rows = []
    for f in FEATURES:
        v = sample[f].to_numpy(dtype=np.float64)
        ok = ~np.isnan(v)
        miss = 1.0 - ok.mean()
        mp = np.nanmean(v[y == 1]) if np.any(ok & (y == 1)) else np.nan
        mn = np.nanmean(v[y == 0]) if np.any(ok & (y == 0)) else np.nan
        rows.append((f, miss, mp, mn, _auc_fast(y, v)))
    rows.sort(key=lambda r: -(abs(r[4] - 0.5) if not np.isnan(r[4]) else -1))
    for f, miss, mp, mn, auc in rows:
        sep = abs(auc - 0.5) if not np.isnan(auc) else np.nan
        log(f"  {f:<28}{miss:>9.3f}{mp:>11.4f}{mn:>11.4f}{auc:>8.4f}{sep:>10.4f}")
    log("  (AUC on non-missing rows only; AUC < 0.5 just means 'lower = more likely a match')")
    dead = [f for f, miss, _, _, auc in rows if miss >= 0.999 or np.isnan(auc)]
    if dead:
        log(f"\n  empty / constant here (the notebook drops these automatically): {dead}")

    sub = sample[FEATURES].sample(min(len(sample), 300_000), random_state=0)
    corr = sub.corr().abs()
    hi = [(a, b, corr.loc[a, b]) for i, a in enumerate(FEATURES)
          for b in FEATURES[i + 1:] if corr.loc[a, b] > CORR_WARN]
    log(f"\n  highly redundant pairs (|pearson| > {CORR_WARN}): {len(hi)}")
    for a, b, c in sorted(hi, key=lambda x: -x[2]):
        log(f"    {a:<28} ~ {b:<28} {c:.3f}")
    if hi:
        log("  -> drop one of each pair if the GBDT gives it ~zero importance "
            "(DROP_FEATURES in the notebook)")


# ============================================================================
#  main
# ============================================================================
def _read_pairs(split, log):
    path = ART / f"candidate_pairs_{split}.tsv"
    header = pd.read_csv(path, sep="\t", nrows=0).columns
    want = ["source1_entity_id", "candidate_entity_id", "label", "evidence_mask"] + list(BLK_MAP)
    use = [c for c in want if c in header]
    dt = {"source1_entity_id": "category", "candidate_entity_id": str}
    t = time.time()
    P = pd.read_csv(path, sep="\t", usecols=use, dtype=dt, keep_default_na=False,
                    na_values={c: [""] for c in use if c not in dt})
    missing_blk = [c for c in list(BLK_MAP) + ["evidence_mask"] if c not in P.columns]
    log(f"  pairs: {len(P):,} rows from {path.name}   [{time.time() - t:.1f}s]")
    if missing_blk:
        log(f"  NOTE: blocking columns {missing_blk} not in file -> those features stay NaN")
    return P


def run(split, log=print):
    t_all = time.time()
    log("#" * 100)
    log(f"#  FEATURE ENGINEERING (final) — split = {split}   ({len(FEATURES)} features)")
    log("#" * 100)
    cfg = FILES[split]
    P = _read_pairs(split, log)

    s1meta = pd.read_csv(ART / cfg["s1"], usecols=["entity_id", "country_clean"],
                         dtype=str, keep_default_na=False)
    s1meta["country_clean"] = s1meta["country_clean"].str.strip()
    s1meta = s1meta.drop_duplicates("entity_id")
    cats = P["source1_entity_id"].cat.categories
    cat_country = s1meta.set_index("entity_id")["country_clean"].reindex(cats).to_numpy(dtype=object)
    codes = P["source1_entity_id"].cat.codes.to_numpy()
    country = np.where(codes >= 0, cat_country[np.maximum(codes, 0)], None)
    bad = pd.isna(country)
    if bad.any():
        log(f"  WARNING: {int(bad.sum()):,} pairs have an S1 id not in {cfg['s1']} -> dropped")
    P["country"] = country

    fold = None
    split_file = ART / "train_val_split.csv"
    if split == "train" and split_file.exists():
        tv = pd.read_csv(split_file, dtype=str).drop_duplicates("entity_id")
        fold_cat = tv.set_index("entity_id")["split"].reindex(cats).to_numpy(dtype=object)
        fold = np.where(codes >= 0, fold_cat[np.maximum(codes, 0)], None)
        P["fold"] = fold

    has_label = "label" in P.columns
    countries = [c for c in pd.unique(P["country"].dropna())]
    log(f"  blocking key families: {len(BLK_FAM)} "
        f"({'from ' + BLK_SRC if BLK_SRC else 'NOT FOUND - generic names'})")
    log(f"  countries found: {countries}   (processed generically — nothing hard-coded)")

    writer = _Writer(ART / f"pair_features_{split}", log)
    sample_parts = []
    frac = min(1.0, REPORT_SAMPLE / max(len(P), 1))
    n_written = 0

    for ctry in countries:
        t_c = time.time()
        sel = np.flatnonzero((P["country"] == ctry).to_numpy())
        Pc = P.iloc[sel].reset_index(drop=True)
        log(f"\n  {ctry}: {len(Pc):,} pairs")

        t = time.time()
        s1r = _load_records([cfg["s1"]], ctry)
        cdr = _load_records(cfg["cd"], ctry)
        n_s1 = len(s1r)
        R = pd.concat([s1r, cdr], ignore_index=True)
        log(f"    records: S1={n_s1:,}  candidates={len(cdr):,}   [load {time.time() - t:.1f}s]")

        i1 = pd.Index(s1r["id"]).get_indexer(Pc["source1_entity_id"].astype(str))
        i2 = pd.Index(cdr["id"]).get_indexer(Pc["candidate_entity_id"])
        keep = (i1 >= 0) & (i2 >= 0)
        if not keep.all():
            log(f"    WARNING: {int((~keep).sum()):,} pairs reference ids missing from the "
                f"{ctry} records (other country / unknown id) -> dropped")
            Pc = Pc.loc[keep].reset_index(drop=True)
            i1, i2 = i1[keep], i2[keep]
        i2 = i2 + n_s1
        del s1r, cdr
        if not len(Pc):
            continue

        S = _build_space(R, n_s1, log)
        del R
        gc.collect()

        n = len(Pc)
        F = np.full((n, len(FEATURES)), np.nan, dtype=np.float32)
        t = time.time()
        for lo in range(0, n, PAIR_CHUNK):
            hi = min(n, lo + PAIR_CHUNK)
            _batch_features(S, i1[lo:hi], i2[lo:hi], F[lo:hi])
        log(f"    pair features: {n:,} pairs   [{time.time() - t:.1f}s]")

        # blocking columns first: the context features rank pairs by p_block
        for src, dst in BLK_MAP.items():
            if src in Pc.columns:
                F[:, FIDX[dst]] = pd.to_numeric(Pc[src], errors="coerce").to_numpy(dtype=np.float32)
        if "evidence_mask" in Pc.columns:
            mk = pd.to_numeric(Pc["evidence_mask"], errors="coerce").to_numpy(dtype=np.float64)
            okm = np.isfinite(mk)
            mi = np.where(okm, mk, 0).astype(np.int64)
            for k, f in enumerate(EVB_FEATS):
                F[:, FIDX[f]] = np.where(okm, (mi >> k) & 1, np.nan).astype(np.float32)

        t = time.time()
        _context_features(S, i1, i2, F)
        log(f"    context features   [{time.time() - t:.1f}s]")

        # write in slices: no full-size copy of the feature matrix
        t = time.time()
        ids1 = Pc["source1_entity_id"].astype(str).to_numpy()
        ids2 = Pc["candidate_entity_id"].to_numpy()
        fold_c = Pc["fold"].fillna("train").to_numpy() if fold is not None else None
        lab_c = (pd.to_numeric(Pc["label"], errors="coerce").fillna(0).astype(np.int8).to_numpy()
                 if has_label else None)
        for lo in range(0, n, WRITE_CHUNK):
            hi = min(n, lo + WRITE_CHUNK)
            out = pd.DataFrame(F[lo:hi], columns=FEATURES, copy=False)
            out.insert(0, "country", ctry)
            out.insert(0, "candidate_entity_id", ids2[lo:hi])
            out.insert(0, "source1_entity_id", ids1[lo:hi])
            if fold_c is not None:
                out["fold"] = fold_c[lo:hi]
            if lab_c is not None:
                out["label"] = lab_c[lo:hi]
                if frac > 0:
                    sample_parts.append(out.sample(frac=frac, random_state=lo))
            writer.write(out)
            del out
        n_written += n
        am = float(np.maximum(F[:, FIDX["addr_missing_s1"]], F[:, FIDX["addr_missing_cand"]]).mean())
        pos = f"  positives {lab_c.mean():.3f}" if lab_c is not None else ""
        log(f"    either address missing: {am:.3f}{pos}   [write {time.time() - t:.1f}s, "
            f"country {time.time() - t_c:.1f}s]")
        del S, F, Pc, i1, i2
        gc.collect()

    writer.close()
    log(f"\n  saved -> {writer.path}   ({n_written:,} rows, {len(FEATURES)} features)")
    if has_label and sample_parts:
        _report(pd.concat(sample_parts, ignore_index=True), log)
    log(f"\n  total {time.time() - t_all:.1f}s")


def main(split="train"):
    for sp in (["train", "test"] if split == "all" else [split]):
        run(sp)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "train")