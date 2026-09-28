"""
blocking_final.py — Stage 2: candidate generation   (v7, recall-first)

Built on blocking.py v6 (same engine, same output file and columns). What is new:

  * 3 more key families (24 in total), aimed at the true pairs v6 never generated:
      fz_addr    typo-proof name x address:  (first name word + first letter of the
                 second) x address word,  and  (phonetic first name word) x (phonetic
                 address word)  ->  guru tredimg ~ guru trading,  chinglepet ~ chengalpettu
      ad_numw    house / unit number x address word (and two numbers x word):
                 finds pairs whose names are unrelated (seyon deals = gildevo)
      ad_skpair  two phonetic address words:  transliterated addresses
  * candidates with little / no address: first name word + initials (and + first two
    letters) of the other words join the lw_pfx3 family  (pacific casnnaris ~ pacific
    cannabis,  cure gll ~ cure grill)
  * DBA / alias names (name_primary / name_alias from preprocessing) are exact-name keys
  * more address and name words per record in the name x address families
  * a wider funnel: bigger pool, bigger shortlist, larger S1 cap
  * the pair budget is chosen for RECALL: the smallest budget reaching TARGET_RECALL
    (or the shortlist's recall), but never below MIN_PRECISION
  * the report adds, for every miss, how many S1 share the candidate's name
    ("tie size") and writes every missed pair to artifacts/blocking_misses_train.csv

The 21 v6 families keep their bit positions (new ones are appended), so
feature_engineering.py decodes evidence_mask correctly with either file.

    STEP 0   country hard filter ................................ recall 1.0   (unchanged)
    STEP 1   17 parallel hash-map blockers + union recall ....... DIAGNOSTIC ONLY
             It never feeds candidate_pairs, so it is opt-in:   python blocking_final.py train step1
    STEP 2   the real candidate generator  ->  artifacts/candidate_pairs_{split}.tsv
             2a  record keys ...... raw and phonetic names (two phonetic levels), name
                                    variants with 1-2 words dropped (both sides), the name
                                    as an order-free word SET, whole-name typo variants,
                                    word prefixes, addresses, house numbers, a 128-bit
                                    character / word signature of every name and address,
                                    and the legal form preprocessing stripped from the name.
                                    Glued web names are split back into words using the S1
                                    vocabulary (servicesprivatenaturecom -> services nature).
                                    Candidate-only noise words are removed from the keys.
             2b  key index ........ 24 composite-key families.  A key is dropped when too
                                    many S1 records share it; a key few S1 records share is
                                    kept even when many candidates do (it yields few pairs
                                    per candidate).  Candidates with little or no address
                                    also get name-only families with much larger caps.
             2c  pairs ............ every candidate x every S1 sharing a key
             2d  pool ............. a learned linear pre-score ranks every generated pair;
                                    each candidate keeps its POOL_K best
             2e  pair features .... IDF-weighted name / address overlap on the FULL records,
                                    character similarity, exact-name / word-set / address /
                                    number / postal-code agreement, name and address ambiguity
             2f  blocking ranker .. LightGBM -> p (a loose shortlist per candidate)
             2g  context re-score . a second small model sees every shortlisted pair in the
                                    context of its candidate AND its S1 (how many other
                                    candidates compete for that S1, how many S1 the candidate
                                    hesitates between) -> p_block
             2h  selection ........ a PAIR BUDGET per country: the highest p_block pairs.
                                    Automatic: the smallest budget whose train recall reaches
                                    min(TARGET_RECALL, shortlist recall - RECALL_SLACK), limited
                                    so precision stays >= MIN_PRECISION (the same budget is
                                    saved for test).

Run:   python blocking_final.py                 train: Step 0 + 2   (fits and saves the models)
       python blocking_final.py train step1     train: Step 0 + 1 + 2
       python blocking_final.py test            test : Step 0 + 2   (uses the models saved by train)

Input: artifacts/{split}_s{1,2,3}_clean.csv from 0_eda_preprocessing_final.ipynb (older
cleaned files also work; the alias keys are then simply absent).

Output columns (unchanged names, read by feature_engineering.py):
    source1_entity_id  candidate_entity_id  score  n_keys  evidence_mask  n_evidence
    cand_rank  p_block  [label]

Nothing is specific to a country: every country in the data is processed the same way.
"""
import concurrent.futures as cf
import gc
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    BASE = Path(__file__).resolve().parent
except NameError:
    BASE = Path(".").resolve()
ART = BASE / "artifacts"

# ============================================================================
#  knobs — STEP 1  (diagnostic only)
# ============================================================================
RUN_STEP1   = False      # default off: Step 1 never changes candidate_pairs
READ_CHUNK  = 1_000_000
PROBE_CHUNK = 1_000_000
NGRAM_NAME  = 4          # char n-gram size on the romanised name
NGRAM_SKEL  = 5          # char n-gram size on the phonetic skeleton
MISS_SAMPLE = 30         # missed pairs to print per country (0 = off)

# Per-blocker generation config:
#   block  = drop a key entirely if its candidate block exceeds this
#   k      = per S1, use at most its k rarest keys
#   df     = index-side document-frequency cap (n-gram blockers only)
#   budget = per S1, stop adding keys once this many candidates are reached
#   minlen = minimum token length (skeleton tokens degenerate below 4)
CFG = {
    "name_exact":  dict(block=50000, k=None, df=None,  budget=None, minlen=None),
    "name_sorted": dict(block=50000, k=None, df=None,  budget=None, minlen=None),
    "name_flat":   dict(block=50000, k=None, df=None,  budget=None, minlen=None),
    "name_token":  dict(block=80000, k=12,   df=None,  budget=700,  minlen=None),
    "name_ngram":  dict(block=60000, k=24,   df=50000, budget=900,  minlen=None),
    "skel_exact":  dict(block=50000, k=None, df=None,  budget=None, minlen=None),
    "skel2_exact": dict(block=50000, k=None, df=None,  budget=None, minlen=None),
    "skel_token":  dict(block=80000, k=12,   df=None,  budget=400,  minlen=4),
    "skel2_token": dict(block=80000, k=12,   df=None,  budget=400,  minlen=3),
    "skel_ngram":  dict(block=60000, k=24,   df=50000, budget=700,  minlen=None),
    "addr_sorted": dict(block=50000, k=None, df=None,  budget=None, minlen=None),
    "addr_pin":    dict(block=50000, k=None, df=None,  budget=None, minlen=None),
    "addr_token":  dict(block=60000, k=12,   df=None,  budget=900,  minlen=None),
    "addr_num":    dict(block=30000, k=8,    df=None,  budget=300,  minlen=None),
    "addr_alnum":  dict(block=30000, k=8,    df=None,  budget=300,  minlen=None),
    "anchor":      dict(block=50000, k=None, df=None,  budget=None, minlen=None),
    "anchor2":     dict(block=50000, k=None, df=None,  budget=None, minlen=None),
}
ORDER = ["name_exact", "name_sorted", "name_flat", "name_token", "name_ngram",
         "skel_exact", "skel2_exact", "skel_token", "skel2_token", "skel_ngram",
         "addr_sorted", "addr_pin", "addr_token", "addr_num", "addr_alnum",
         "anchor", "anchor2"]
ENABLED = {b: True for b in ORDER}

# ============================================================================
#  knobs — STEP 2
# ============================================================================
N_WORKERS = 1            # countries processed in parallel. Each worker holds one country
                         # in memory (~5-7 GB at full scale): use 2 only with >= 16 GB RAM
                         # and nothing else running.

# ---- key families:  (cap_s1, cap_cand, weight, key-space parts) ----
#   cap_s1   a key shared by more S1 records than this is too generic -> dropped
#   cap_cand ... or by more candidates than this -- unless few S1 records share it:
#            a key shared by k S1 records may reach min(CC_BOOST, cap_s1 / k) x cap_cand
#            candidates (each of them gets only k pairs from it)
#   weight   multiplies the key's rarity weight in the cheap score
#   parts    key-space partitions while counting (memory only)
# nm = raw name (after digit look-alike / web-noise fixes),  sk = phonetic skeleton
# F = full name, 1 = one word dropped, S = one or two words dropped on the S1 side,
# F1 = one or two words dropped on the candidate side
FAMILIES = {
    "nmF_FF":  (30, 3000, 2.0, 1),   # exact name                      (fevaex = fevaex)
    "nmF_SF":  (20, 3000, 1.3, 1),   # candidate lacks 1-2 S1 words     (capital harbor = capital harbor artificial)
    "nmF_F1":  (20, 3000, 1.3, 1),   # candidate has 1-2 extra words    (willow industries = willow)
    "nmF_11":  (12, 1500, 0.9, 1),   # one word differs                 (smith virtus services ~ smith rocky virtus)
    "skF_FF":  (40, 3000, 1.6, 1),   # same, phonetic skeleton          (laksmi = lakshmi, hai keyara = high care)
    "skF_SF":  (20, 3000, 1.1, 1),
    "skF_11":  (12, 1500, 0.8, 1),
    "nm_set":  (30, 3000, 1.5, 1),   # same words, any order; S1 may have 1-2 more
                                     #   (servicesprivatenaturecom = nature services, fordmcneilcom = mcneil and ford ...)
    "nm_pfx":  (25, 3000, 1.0, 1),   # first PFX_LEN letters of the name, spaces removed (domains)
    "nm_char": (15, 1500, 1.0, 1),   # whole name within one typo       (olanent staed = olanent staked)
    "nm_tok":  (15, 1500, 0.8, 1),   # one rare name word
    "nm_addr": (40,  400, 1.0, 2),   # rarest name word x address word
    "fl_addr": (30, 1000, 1.5, 3),   # phonetic whole name (or name minus 1-2 words) x address word
    "rw_addr": (30, 1000, 1.4, 3),   # raw whole name (or name minus 1 word) x address word
                                     #   (rare raw words whose phonetic form is generic: patodia -> bd)
    "s2F_FF":  (40, 3000, 1.3, 1),   # coarse phonetic name, nasals dropped (international = imtarnyasanal)
    "ad_full": (20, 1000, 2.0, 1),   # whole address
    "ad_pair": (50,  400, 1.0, 3),   # two address words
    "ad_numx": (30, 1000, 1.0, 1),   # house number minus its last digit x address word  (2146 ~ 2148 yuma)
    # candidates with little or no address (LOW_ADDR_W): the name is all they have,
    # so their name keys may be shared by many more S1 records
    "lw_name": (300, 5000, 1.0, 1),  # raw name: full / minus 1-2 words, both sides (+ alias names)
    "lw_skel": (300, 5000, 0.9, 1),  # same, phonetic skeleton
    "lw_pfx3": (300, 5000, 0.8, 1),  # first 3 letters of every word    (gujarat constsouctoin ~ gujarat construction)
                                     # + first word & initials / first 2 letters of the others
                                     #   (pacific casnnaris ~ pacific cannabis, cure gll ~ cure grill)
    # ---- v7 (appended: the v6 bit positions above are unchanged) ----
    "fz_addr":   (30,  800, 1.2, 3),  # typo-proof name x address word:
                                      #   (first word + first letter of the 2nd) x address word
                                      #   (phonetic first word) x (phonetic address word)
    "ad_numw":   (30,  600, 1.0, 3),  # number x address word, two numbers x address word
                                      #   (unrelated names at the same address:  15 16 x indore)
    "ad_skpair": (40,  600, 0.9, 2),  # two phonetic address words (chinglepet ~ chengalpettu)
}
FAM = list(FAMILIES)                  # bit i of evidence_mask = FAM[i]
NF = len(FAM)
assert NF <= 24, "evidence tables are 2^NF entries: keep NF <= 24"
STRONG = ("nmF_FF", "skF_FF", "nm_set", "ad_full")   # these always enter the pool (up to POOL_STRONG)
CC_BOOST = 10                  # see cap_cand above

# ---- per-record key budgets:  (S1 side, candidate side) ----
# The S1 side (long, complete records) keys on more of its words than the
# candidate side (short, partial records), so a partial candidate still meets
# the S1 record it came from.
W_TOK      = 6                 # name words used for name variants (longer names: first 6)
DEL_MAX    = 6                 # one-word-dropped variants for names of <= this many words
DEL2_MAX   = 5                 # two-words-dropped variants for names of 4..5 words (both sides)
FA_MAX     = 5                 # fl_addr: name variants only for names of <= this many words
PFX_LEN    = 10                # nm_pfx prefix length (names longer than this only)
PFX3       = 3                 # lw_pfx3: letters kept per word
CD_LEN     = (4, 22)           # nm_char: names of this many letters (spaces removed)
K_TOK      = (2, 1)            # nm_tok (candidate): rarest raw words, rarest skeleton words
K_NA       = ((2, 12), (1, 5)) # nm_addr: (name words, address words) for S1 / candidate
K_FA       = (16, 5)           # fl_addr: address words
K_AP       = (10, 5)           # ad_pair: rarest address words
K_NX       = (8, 4)            # ad_numx: rarest address words
K_FZ       = ((12, 8), (5, 4)) # fz_addr: (address words, phonetic address words) S1 / candidate
K_NW       = ((3, 12), (3, 5)) # ad_numw: (numbers, non-number address words)     S1 / candidate
K_SP       = (6, 3)            # ad_skpair: rarest phonetic address words
FWI_N      = 4                 # lw_pfx3: first word + this many initials / 2-letter prefixes
SKA_MIN    = 3                 # phonetic address words shorter than this are ignored

LOW_ADDR_W = 0.8               # candidate address "low" if its words' summed rarity < this
NOISE_LEN   = 3                # candidate-only noise words: at most this many letters ...
NOISE_DF    = 100              # ... in at least this many candidates ...
NOISE_RATIO = 10.0             # ... and this many times more frequent than among S1 names

# glued web names (candidate words ending in com / net / org that no S1 name has) are
# split into S1 name words + legal-form words; legal-form words = words of the full S1
# names (name_translit) that never occur in name_core, in >= SEG_LEGAL_DF S1 names
SEG_MIN     = 5                # stem length (letters before com / net / org)
SEG_MAXW    = 24               # longest word tried
SEG_LEGAL_DF = 20

S2_BATCH    = 5_000_000        # pair postings per candidate batch (memory only)
S2_READ     = 500_000          # rows per CSV chunk (memory only)
S2_MAX_POST = 3_000_000_000    # runtime guard: max pair postings per country
S2_THREADS  = 3                # candidate batches scored in parallel (numpy / LightGBM release
                               # the GIL).  Each thread holds one batch (~0.5 GB): use 1-2 on
                               # an 8 GB machine.
IDX_THREADS = 2                # key families counted in parallel in 2b

# ---- pool + selection ----
POOL_K        = 20             # rich features + ranker on each candidate's POOL_K best pairs ...
POOL_K_LOW    = 100            # ... (POOL_K_LOW for a candidate with a low / no address)
POOL_STRONG   = 40             # ... plus any STRONG pair ranked below POOL_STRONG
KC_MAX        = 30             # shortlist: a candidate keeps at most this many S1 ...
KC_MAX_LOW    = 60             # ... (this many with a low / no address) ...
P_FLOOR       = 0.0002         # ... and never a pair below this p (unless it is its best)
TARGET_PER_S1 = None           # PAIR BUDGET per S1.  None = automatic (see below);
                               # a number (e.g. 6.0) forces that budget on train and test
TARGET_RECALL = 0.999          # automatic budget: smallest budget reaching this recall ...
RECALL_SLACK  = 0.0001         # ... or the shortlist's recall minus this, if that is lower ...
MIN_PRECISION = 0.50           # ... but never a budget whose precision falls below this
MAX_PER_S1    = 12.0           # the automatic budget never exceeds this
DEFAULT_PER_S1 = 6.0           # test without a saved budget
AUTO_GRID     = [round(3.0 + 0.1 * i, 1) for i in range(91)]
TOPK_S1       = 100            # an S1 entity keeps at most this many candidates
MISS_CSV      = True           # train: every missed true pair -> artifacts/blocking_misses_train.csv

# ---- learned pre-score (pool ranking) + blocking ranker + context re-score ----
PRE_EVERY     = 5              # pre-score fitted on every 5th candidate batch ...
PRE_SAMPLE    = 1_500_000      # ... this many generated pairs per country (all positives kept)
PRE_POOLED    = 500_000        # per-country rows used for the pre-score saved for test
RANKER_SAMPLE = 1_000_000      # labelled pool pairs per country to fit the ranker (train)
RANKER_EVERY  = 4              # ... drawn from every 4th candidate batch
RANKER_POOLED = 600_000        # per-country rows used for the model saved for test
STACK_SAMPLE  = 2_000_000      # shortlisted pairs per fold to fit the context re-score (train)
STACK_POOLED  = 800_000        # per-country rows used for the re-score model saved for test
RANKER_PATH   = ART / "models" / "blocking_final_ranker.joblib"

POOLK_SWEEP  = [1, 2, 3, 5, 8, 12, 20, 30, 40]
BUDGET_SWEEP = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 10.0, 12.0]


COLS = ["entity_id", "country_clean", "business_name", "name_core",
        "name_tokens_sorted", "address_tokens_sorted", "address_pin"]

FILES = {
    "train": dict(s1="train_s1_clean.csv",
                  cd=["train_s2_clean.csv", "train_s3_clean.csv"],
                  gt="ground_truth_clean.csv"),
    "test":  dict(s1="test_s1_clean.csv",
                  cd=["test_s2_clean.csv", "test_s3_clean.csv"],
                  gt=None),
}

# ============================================================================
#  byte-level text engine
#  code alphabet: 0=separator  1=space  2..27=a-z  28..37=0-9  38=other
# ============================================================================
def _base_lut():
    l = np.full(256, 38, dtype=np.uint8)
    l[ord("a"):ord("z") + 1] = np.arange(2, 28)
    l[ord("0"):ord("9") + 1] = np.arange(28, 38)
    l[ord(" ")] = 1
    l[1] = 0
    return l

_LUT = _base_lut()
_ALPHA = 39
_POW = np.array([pow(1000003, i, 2 ** 64) for i in range(32)], dtype=np.uint64)


def _skel_lut():
    """phonetic skeleton: drop vowels/h/y, collapse confusable consonants.
    silver -> slbr , jhilvar -> slbr , private -> brbd , praiveta -> brbd"""
    l = np.arange(39, dtype=np.uint8)
    o = lambda c: ord(c) - ord("a") + 2
    for c in "aeiouhy":
        l[o(c)] = 255
    l[38] = 255
    for c in "vwpf":
        l[o(c)] = o("b")
    for c in "szjx":
        l[o(c)] = o("s")
    l[o("c")] = o("s")
    l[o("t")] = o("d")
    for c in "gq":
        l[o(c)] = o("k")
    return l

_SKEL = _skel_lut()


def _encode(texts):
    return _LUT[np.frombuffer(("\x01".join(texts) + "\x01").encode("ascii", "ignore"),
                              dtype=np.uint8)]


def _tidy_spaces(s):
    """drop leading / trailing / repeated spaces inside each record."""
    if s.size == 0:
        return s
    prev = np.concatenate(([np.uint8(0)], s[:-1]))
    s = s[~((s == 1) & ((prev == 1) | (prev == 0)))]
    if s.size == 0:
        return s
    nxt = np.concatenate((s[1:], [np.uint8(0)]))
    return s[~((s == 1) & (nxt == 0))]


def _skeletonize(code):
    s = _SKEL[code]
    return _tidy_spaces(s[s != 255])


def _flatten(code):
    z = _skeletonize(code)
    return z[z != 1]


def _rowids(code):
    return np.cumsum(code == 0, dtype=np.int32)


def _ngrams_from_code(code, n, base_row, keep=None):
    if code.size <= n:
        return np.empty(0, np.int32), np.empty(0, np.int64)
    row = _rowids(code)
    g = code[:-(n - 1)].astype(np.int64)
    ok = code[:-(n - 1)] != 0
    for k in range(1, n):
        tail = code[k:len(code) - (n - 1 - k)]
        g = g * _ALPHA + tail
        ok &= tail != 0
    r, g = row[:-(n - 1)][ok], g[ok]
    if keep is not None:
        m = keep[g]
        r, g = r[m], g[m]
    space = _ALPHA ** n
    u = np.unique(r.astype(np.int64) * space + g)
    return (u // space).astype(np.int32) + base_row, (u % space).astype(np.int64)


def _tokens_from_code(code, base_row):
    """(row, token_hash, length, has_digit, all_digit) — no Python strings."""
    is_tok = code >= 2
    if not is_tok.any():
        return (np.empty(0, np.int32), np.empty(0, np.uint64),
                np.empty(0, np.int16), np.empty(0, bool), np.empty(0, bool))
    prev = np.concatenate(([False], is_tok[:-1]))
    nxt = np.concatenate((is_tok[1:], [False]))
    starts = np.flatnonzero(is_tok & ~prev)
    ends = np.flatnonzero(is_tok & ~nxt) + 1
    lens = (ends - starts).astype(np.int64)

    chars = code[is_tok].astype(np.uint64)
    pos = np.flatnonzero(is_tok)
    off = np.minimum(pos - np.repeat(starts, lens), 31).astype(np.int64)
    seg = np.concatenate(([0], np.cumsum(lens)[:-1]))
    h = np.add.reduceat(chars * _POW[off], seg)
    h = h * np.uint64(1000003) + lens.astype(np.uint64)

    dig = ((chars >= 28) & (chars <= 37)).astype(np.int64)
    nd = np.add.reduceat(dig, seg)
    rows = _rowids(code)[starts].astype(np.int32) + base_row
    return rows, h, lens.astype(np.int16), nd > 0, nd == lens


def _chunked(texts, fn):
    outs = None
    for st in range(0, len(texts), READ_CHUNK):
        res = fn(_encode(texts[st:st + READ_CHUNK]), st)
        if outs is None:
            outs = [[r] for r in res]
        else:
            for i, r in enumerate(res):
                outs[i].append(r)
    if outs is None:
        return None
    return [np.concatenate(o) for o in outs]


# ============================================================================
#  small helpers
# ============================================================================
def _compact(a, b):
    both = np.concatenate([a.astype(np.uint64), b.astype(np.uint64)])
    _, inv = np.unique(both, return_inverse=True)
    inv = inv.astype(np.int64).ravel()
    return inv[:len(a)], inv[len(a):]


def _dedupe(rows, keys):
    if len(keys) == 0:
        return rows.astype(np.int32), keys.astype(np.int64)
    nk = int(keys.max()) + 1
    u = np.unique(rows.astype(np.int64) * nk + keys)
    return (u // nk).astype(np.int32), (u % nk).astype(np.int64)


def _rarest_k(rows, keys, df, k):
    if k is None or len(rows) == 0:
        return rows, keys
    o = np.lexsort((df[keys], rows))
    rows, keys = rows[o], keys[o]
    n = int(rows.max()) + 1
    start = np.searchsorted(rows, np.arange(n))
    m = (np.arange(len(rows), dtype=np.int64) - start[rows]) < k
    return rows[m], keys[m]


def _rarest_one(rows, keys, df, n_rows):
    r, k = _dedupe(rows, keys)
    r, k = _rarest_k(r, k, df, 1)
    out = np.full(n_rows, -1, dtype=np.int64)
    out[r] = k
    return out


def _expand(starts, counts):
    """concatenate ranges [starts[i], starts[i]+counts[i]) — fully vectorised."""
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, np.int64)
    return (np.arange(total, dtype=np.int64)
            - np.repeat(np.cumsum(counts) - counts, counts)
            + np.repeat(np.asarray(starts, dtype=np.int64), counts))


def _group_starts(sorted_ids, n):
    """first index of each id in a sorted id array (length n+1, CSR style)."""
    off = np.zeros(n + 1, dtype=np.int64)
    off[1:] = np.cumsum(np.bincount(sorted_ids, minlength=n))
    return off


def _in_sorted(sorted_arr, q):
    """boolean: is each q in sorted_arr."""
    if sorted_arr.size == 0 or q.size == 0:
        return np.zeros(q.size, bool)
    p = np.searchsorted(sorted_arr, q)
    return (p < sorted_arr.size) & (sorted_arr[np.minimum(p, sorted_arr.size - 1)] == q)


def _apply_caps(s_sid, s_key, c_cid, c_key, n_s1, cf):
    """STEP 1 caps — exactly as in the 0.9937 run: global block cap, then per
    S1 keys are taken rarest-first while the running total BEFORE the key is
    still below the budget."""
    if len(s_key) == 0 or len(c_key) == 0:
        return None
    nk = int(max(s_key.max(), c_key.max())) + 1
    n1 = np.bincount(s_key, minlength=nk)
    n2 = np.bincount(c_key, minlength=nk)

    live = (n1 > 0) & (n2 > 0) & (n2 <= cf["block"])
    s_m, c_m = live[s_key], live[c_key]
    s_sid, s_key = s_sid[s_m], s_key[s_m]
    c_cid, c_key = c_cid[c_m], c_key[c_m]
    n_keys = int(live.sum())
    if len(s_key) == 0 or len(c_key) == 0:
        return None

    if cf["budget"]:
        o = np.lexsort((n2[s_key], s_sid))           # rarest key first per S1
        s_sid, s_key = s_sid[o], s_key[o]
        sz = n2[s_key].astype(np.int64)
        csum = np.cumsum(sz)
        start = np.searchsorted(s_sid, np.arange(n_s1))
        base = np.concatenate(([0], csum))[start][s_sid]
        keep = (csum - sz - base) < cf["budget"]     # total BEFORE this key
        s_sid, s_key = s_sid[keep], s_key[keep]
        if len(s_key) == 0:
            return None

    n_pairs = int(n2[s_key].astype(np.int64).sum())
    return s_sid, s_key, c_cid, c_key, n1, n2, nk, n_keys, n_pairs


def _probe_recall(s_sid, s_key, c_cid, c_key, nk, t_sid, t_cid, T, n_s1):
    """which ground-truth pairs this blocker would produce — no pairs built."""
    c_code = np.sort(c_cid.astype(np.int64) * nk + c_key)
    o = np.argsort(s_sid, kind="stable")
    ssid, skey = s_sid[o], s_key[o]
    cnt = np.bincount(ssid, minlength=n_s1)
    start = np.concatenate(([0], np.cumsum(cnt)[:-1])).astype(np.int64)

    mask = np.zeros(T, bool)
    for lo in range(0, T, PROBE_CHUNK):
        hi = min(lo + PROBE_CHUNK, T)
        ts = t_sid[lo:hi]
        rep = cnt[ts]
        tot = int(rep.sum())
        if tot == 0:
            continue
        p = np.repeat(np.arange(lo, hi, dtype=np.int64), rep)
        k = skey[_expand(start[ts], rep)]
        probe = t_cid[p].astype(np.int64) * nk + k
        mask[p[_in_sorted(c_code, probe)]] = True
    return mask


# ============================================================================
#  loading
# ============================================================================
def _read_country(path, country):
    parts = []
    for ch in pd.read_csv(ART / path, usecols=COLS, dtype={c: str for c in COLS},
                          chunksize=READ_CHUNK):
        parts.append(ch[ch["country_clean"] == country])
    df = pd.concat(parts, ignore_index=True)
    for c in COLS:
        df[c] = df[c].fillna("")
    return df


def build_truth(gt, log):
    matched = gt.dropna(subset=["matched_entity_ids"])
    t = matched.copy()
    t["cand"] = t["matched_entity_ids"].str.split(",")
    t = t.explode("cand")
    t["cand"] = t["cand"].str.strip()
    t = t.rename(columns={"source1_entity_id": "s1"})[["s1", "cand"]].reset_index(drop=True)
    log(f"  ground truth: {len(t):,} true pairs from {len(matched):,} matched S1 entities"
        f"   |   {len(gt) - len(matched):,} singletons")
    return t


def _truth_local(truth, s1c, candc):
    """ground truth for one country as local row indices (s1_row, cand_row)."""
    n1, n2 = len(s1c), len(candc)
    ts = truth["s1"].map(pd.Series(np.arange(n1, dtype=np.int64),
                                   index=s1c["entity_id"].to_numpy())).to_numpy()
    tcd = truth["cand"].map(pd.Series(np.arange(n2, dtype=np.int64),
                                      index=candc["entity_id"].to_numpy())).to_numpy()
    ok = ~(pd.isna(ts) | pd.isna(tcd))
    return ts[ok].astype(np.int64), tcd[ok].astype(np.int64)


# ============================================================================
#  STEP 0 — country hard filter
# ============================================================================
def step0(split, truth, log):
    t0 = time.time()
    log("\n" + "=" * 100)
    log("STEP 0 — COUNTRY HARD FILTER")
    log("=" * 100)
    cfg = FILES[split]
    s1 = pd.read_csv(ART / cfg["s1"], usecols=["entity_id", "country_clean"], dtype=str)
    cd = pd.concat([pd.read_csv(ART / f, usecols=["entity_id", "country_clean"], dtype=str)
                    for f in cfg["cd"]], ignore_index=True)

    n1c, n2c = s1["country_clean"].value_counts(), cd["country_clean"].value_counts()
    countries = sorted(set(n1c.index) | set(n2c.index))

    hdr = (f"  {'country':<9}{'S1':>12}{'candidates':>13}{'true pairs':>13}"
           f"{'recall':>9}{'precision':>12}{'space':>12}")
    log(hdr); log("  " + "-" * (len(hdr) - 2))

    t = None
    if truth is not None:
        t = truth.copy()
        t["s1_country"] = t["s1"].map(pd.Series(s1["country_clean"].to_numpy(),
                                                index=s1["entity_id"].to_numpy()))
        t["cd_country"] = t["cand"].map(pd.Series(cd["country_clean"].to_numpy(),
                                                  index=cd["entity_id"].to_numpy()))
        t["present"] = t["cd_country"].notna()
        t["kept"] = t["present"] & (t["s1_country"] == t["cd_country"])

    space = 0
    for c in countries:
        a, b = int(n1c.get(c, 0)), int(n2c.get(c, 0))
        space += a * b
        if t is not None:
            sub = t[t["s1_country"] == c]
            n, k = len(sub), int(sub["kept"].sum())
            log(f"  {c:<9}{a:>12,}{b:>13,}{n:>13,}{k/max(n,1):>9.4f}"
                f"{k/max(a*b,1):>12.2e}{a*b:>12.2e}")
        else:
            log(f"  {c:<9}{a:>12,}{b:>13,}{'n/a':>13}{'n/a':>9}{'n/a':>12}{a*b:>12.2e}")
    log("  " + "-" * (len(hdr) - 2))

    if t is not None:
        T, K = len(t), int(t["kept"].sum())
        log(f"  {'ALL':<9}{len(s1):>12,}{len(cd):>13,}{T:>13,}{K/T:>9.4f}"
            f"{K/space:>12.2e}{space:>12.2e}")
        miss = int((~t["present"]).sum())
        cross = int((t["present"] & (t["s1_country"] != t["cd_country"])).sum())
        log("")
        log(f"  candidate id absent from S2 u S3 ... {miss:,} ({miss/T:.4%})  <- unreachable, caps all recall")
        log(f"  cross-country true pairs ........... {cross:,} ({cross/T:.4%})  <- destroyed by this filter")
        log(f"  STEP 0 RECALL    = {K/T:.6f}")
        log(f"  STEP 0 PRECISION = {K/space:.3e}   (1 true pair per {space/max(K,1):,.0f} candidates)")
    else:
        log(f"  {'ALL':<9}{len(s1):>12,}{len(cd):>13,}{'n/a':>13}{'n/a':>9}{'n/a':>12}{space:>12.2e}")

    log(f"  space {len(s1)*len(cd):.2e} -> {space:.2e}  ({len(s1)*len(cd)/space:.2f}x)"
        f"   [{time.time()-t0:.1f}s]")
    del s1, cd, t
    gc.collect()
    return countries, space


# ============================================================================
#  STEP 1 — blocker construction
# ============================================================================
def _make_builder(s1c, candc, n1, n2):
    """Returns build(name) -> (s1_rows, s1_keys, cand_rows, cand_keys).
    The byte-engine passes are computed once and reused by every blocker."""
    nm_s = s1c["name_core"].tolist();             nm_c = candc["name_core"].tolist()
    ad_s = s1c["address_tokens_sorted"].tolist(); ad_c = candc["address_tokens_sorted"].tolist()

    ntok_s  = _chunked(nm_s, lambda c, b: _tokens_from_code(c, b))
    ntok_c  = _chunked(nm_c, lambda c, b: _tokens_from_code(c, b))
    atok_s  = _chunked(ad_s, lambda c, b: _tokens_from_code(c, b))
    atok_c  = _chunked(ad_c, lambda c, b: _tokens_from_code(c, b))
    sktok_s = _chunked(nm_s, lambda c, b: _tokens_from_code(_skeletonize(c), b))
    sktok_c = _chunked(nm_c, lambda c, b: _tokens_from_code(_skeletonize(c), b))
    skflat_s = _chunked(nm_s, lambda c, b: _tokens_from_code(_flatten(c), b))
    skflat_c = _chunked(nm_c, lambda c, b: _tokens_from_code(_flatten(c), b))

    def _flat_raw(c, b):                     # whole name, spaces removed (edgeland = edge land)
        z = _name_clean(c)
        return _tokens_from_code(z[z != 1], b)

    def _flat_sk2(c, b):                     # transliteration-aware skeleton, spaces removed
        z = _skel2(_name_clean(c), 1)
        z = z[z != 1]
        if z.size:
            z = z[~np.concatenate(([False], (z[1:] == z[:-1]) & (z[1:] >= 2)))]
        return _tokens_from_code(z, b)

    nflat_s = _chunked(nm_s, _flat_raw);   nflat_c = _chunked(nm_c, _flat_raw)
    sk2flat_s = _chunked(nm_s, _flat_sk2); sk2flat_c = _chunked(nm_c, _flat_sk2)
    sk2tok_s = _chunked(nm_s, lambda c, b: _tokens_from_code(_skel2(_name_clean(c), 1), b))
    sk2tok_c = _chunked(nm_c, lambda c, b: _tokens_from_code(_skel2(_name_clean(c), 1), b))

    def build(b):
        cf = CFG[b]

        # ---- single-key string blockers ----
        if b in ("name_exact", "name_sorted", "addr_sorted", "addr_pin", "anchor"):
            if b == "name_exact":
                sv, cv = s1c["name_core"].to_numpy(), candc["name_core"].to_numpy()
            elif b == "name_sorted":
                sv, cv = s1c["name_tokens_sorted"].to_numpy(), candc["name_tokens_sorted"].to_numpy()
            elif b == "addr_sorted":
                sv, cv = s1c["address_tokens_sorted"].to_numpy(), candc["address_tokens_sorted"].to_numpy()
            elif b == "addr_pin":
                sv, cv = s1c["address_pin"].to_numpy(), candc["address_pin"].to_numpy()
            else:                                    # anchor = name prefix + PIN prefix
                def anc(d):
                    a = d["name_tokens_sorted"].str.replace(" ", "", regex=False).str[:4]
                    p = d["address_pin"].str[:3]
                    return (a + "|" + p).where((a.str.len() == 4) & (p.str.len() == 3), "").to_numpy()
                sv, cv = anc(s1c), anc(candc)
            sm, cm = sv != "", cv != ""
            codes, _ = pd.factorize(np.concatenate([sv[sm], cv[cm]]))
            ns = int(sm.sum())
            return (np.nonzero(sm)[0].astype(np.int32), codes[:ns].astype(np.int64),
                    np.nonzero(cm)[0].astype(np.int32), codes[ns:].astype(np.int64))

        # ---- whole name as one key (skeleton / raw / transliteration-aware skeleton) ----
        if b in ("skel_exact", "name_flat", "skel2_exact"):
            src_s, src_c = {"skel_exact": (skflat_s, skflat_c), "name_flat": (nflat_s, nflat_c),
                            "skel2_exact": (sk2flat_s, sk2flat_c)}[b]
            if src_s is None or src_c is None:
                return None
            sk, ck = _compact(src_s[1], src_c[1])
            return src_s[0], sk, src_c[0], ck

        # ---- token-family blockers ----
        if b in ("name_token", "skel_token", "skel2_token", "addr_token", "addr_num", "addr_alnum"):
            src = {"name_token": (ntok_s, ntok_c), "skel_token": (sktok_s, sktok_c),
                   "skel2_token": (sk2tok_s, sk2tok_c),
                   "addr_token": (atok_s, atok_c), "addr_num": (atok_s, atok_c),
                   "addr_alnum": (atok_s, atok_c)}[b]
            (sr, sh, sl, sd, sa), (cr, ch_, cl, cd_, ca) = src
            if b == "addr_num":                      # pure numbers: house / plot / PIN
                ms = sa & (sl >= 3) & (sl <= 6); mc = ca & (cl >= 3) & (cl <= 6)
            elif b == "addr_alnum":                  # mixed tokens: f17, b4, 225vivek
                ms = sd & ~sa & (sl >= 2);      mc = cd_ & ~ca & (cl >= 2)
            else:
                ms = np.ones(len(sr), bool);    mc = np.ones(len(cr), bool)
            if cf["minlen"]:
                ms = ms & (sl >= cf["minlen"]); mc = mc & (cl >= cf["minlen"])

            sk, ck = _compact(sh[ms], ch_[mc])
            s_sid, s_key = _dedupe(sr[ms], sk)
            c_cid, c_key = _dedupe(cr[mc], ck)
            if len(s_key) == 0 or len(c_key) == 0:
                return None
            nk = int(max(s_key.max(), c_key.max())) + 1
            df = np.bincount(c_key, minlength=nk)
            if cf["df"]:
                ok = df <= cf["df"]
                s_m, c_m = ok[s_key], ok[c_key]
                s_sid, s_key = s_sid[s_m], s_key[s_m]
                c_cid, c_key = c_cid[c_m], c_key[c_m]
            s_sid, s_key = _rarest_k(s_sid, s_key, df, cf["k"])
            return s_sid, s_key, c_cid, c_key

        # ---- character n-gram blockers ----
        if b in ("name_ngram", "skel_ngram"):
            n = NGRAM_NAME if b == "name_ngram" else NGRAM_SKEL
            prep = (lambda c: c) if b == "name_ngram" else _skeletonize
            space = _ALPHA ** n
            df = np.zeros(space, np.int64)                       # pass 1: doc frequency
            for st in range(0, len(nm_c), READ_CHUNK):
                _, g = _ngrams_from_code(prep(_encode(nm_c[st:st + READ_CHUNK])), n, st)
                df += np.bincount(g, minlength=space)
            keepg = (df > 0) & (df <= cf["df"])                  # pass 2: filtered postings
            c_cid, c_key = _chunked(nm_c, lambda c, bs: _ngrams_from_code(prep(c), n, bs, keepg))
            s_sid, s_key = _chunked(nm_s, lambda c, bs: _ngrams_from_code(prep(c), n, bs, keepg))
            s_sid, s_key = _rarest_k(s_sid, s_key, df, cf["k"])
            return s_sid, s_key, c_cid, c_key

        # ---- composite: rarest skeleton-name token x rarest address token ----
        if b == "anchor2":
            sk_s, sk_c = _compact(sktok_s[1], sktok_c[1])
            ak_s, ak_c = _compact(atok_s[1], atok_c[1])
            if len(sk_c) == 0 or len(ak_c) == 0:
                return None
            df_sk = np.bincount(sk_c, minlength=int(max(sk_s.max(initial=0), sk_c.max(initial=0))) + 1)
            df_ak = np.bincount(ak_c, minlength=int(max(ak_s.max(initial=0), ak_c.max(initial=0))) + 1)
            s_sk = _rarest_one(sktok_s[0], sk_s, df_sk, n1)
            s_ak = _rarest_one(atok_s[0],  ak_s, df_ak, n1)
            c_sk = _rarest_one(sktok_c[0], sk_c, df_sk, n2)
            c_ak = _rarest_one(atok_c[0],  ak_c, df_ak, n2)
            sm = (s_sk >= 0) & (s_ak >= 0)
            cm = (c_sk >= 0) & (c_ak >= 0)
            if not sm.any() or not cm.any():
                return None
            s_raw = s_sk[sm] * 1000003 + s_ak[sm]
            c_raw = c_sk[cm] * 1000003 + c_ak[cm]
            sk, ck = _compact(s_raw.astype(np.uint64), c_raw.astype(np.uint64))
            return (np.nonzero(sm)[0].astype(np.int32), sk,
                    np.nonzero(cm)[0].astype(np.int32), ck)
        return None

    return build


# ============================================================================
#  STEP 1 — per-blocker recall / precision   (+ STEP A miss analysis)
# ============================================================================
def step1(split, countries, truth, log, space0):
    log("\n" + "=" * 100)
    log("STEP 1 — PARALLEL HASH-MAP BLOCKERS  (recall + precision per blocker, then union)")
    log("=" * 100)

    agg = {b: {"found": 0, "pairs": 0} for b in ORDER if ENABLED[b]}
    union_found = total_true = total_s1 = 0
    cfg = FILES[split]

    for ctry in countries:
        tc0 = time.time()
        s1c = _read_country(cfg["s1"], ctry)
        candc = pd.concat([_read_country(f, ctry) for f in cfg["cd"]], ignore_index=True)
        n1, n2 = len(s1c), len(candc)
        total_s1 += n1

        t_sid, t_cid = _truth_local(truth, s1c, candc)
        T = len(t_sid)
        total_true += T

        log(f"\n  {ctry}:  S1={n1:,}  candidates={n2:,}  true pairs={T:,}")
        hdr = (f"    {'blocker':<12}{'keys':>10}{'S1 post':>12}{'cand post':>13}"
               f"{'pairs':>14}{'pairs/S1':>10}{'recall':>9}{'precision':>12}{'union':>9}")
        log(hdr); log("    " + "-" * (len(hdr) - 4))
        union = np.zeros(T, bool)

        build = _make_builder(s1c, candc, n1, n2)
        for b in ORDER:
            if not ENABLED[b]:
                continue
            try:
                built = build(b)
                capped = _apply_caps(*built, n1, CFG[b]) if built is not None else None
            except Exception as e:
                log(f"    {b:<12} SKIPPED ({type(e).__name__}: {e})")
                continue
            if capped is None:
                log(f"    {b:<12} SKIPPED (no postings)")
                continue
            n_s_post, n_c_post = len(built[1]), len(built[3])
            s_sid, s_key, c_cid, c_key, _, _, nk, n_keys, n_pairs = capped
            mask = _probe_recall(s_sid, s_key, c_cid, c_key, nk, t_sid, t_cid, T, n1)
            union |= mask
            f = int(mask.sum())
            agg[b]["found"] += f
            agg[b]["pairs"] += n_pairs
            log(f"    {b:<12}{n_keys:>10,}{n_s_post:>12,}{n_c_post:>13,}"
                f"{n_pairs:>14,}{n_pairs/max(n1,1):>10,.1f}{f/max(T,1):>9.4f}"
                f"{f/max(n_pairs,1):>12.2e}{union.sum()/max(T,1):>9.4f}")
            del s_sid, s_key, c_cid, c_key, mask, capped, built
            gc.collect()

        union_found += int(union.sum())
        log("    " + "-" * (len(hdr) - 4))
        log(f"    {'UNION':<12}{'':>10}{'':>12}{'':>13}{'':>14}{'':>10}"
            f"{union.sum()/max(T,1):>9.4f}{'':>12}{'':>9}   [{time.time()-tc0:.1f}s]")

        if MISS_SAMPLE:
            _miss_report(ctry, union, t_sid, t_cid, s1c, candc, log)
        del s1c, candc, build
        gc.collect()

    log("\n" + "=" * 100)
    log("STEP 1 — OVERALL (all countries)")
    log("=" * 100)
    hdr = f"  {'blocker':<13}{'recall':>10}{'precision':>13}{'pairs':>17}{'pairs/S1':>12}"
    log(hdr); log("  " + "-" * (len(hdr) - 2))
    for b in ORDER:
        if not ENABLED[b]:
            continue
        p, f = agg[b]["pairs"], agg[b]["found"]
        log(f"  {b:<13}{f/total_true:>10.4f}{f/max(p,1):>13.2e}{p:>17,}{p/max(total_s1,1):>12,.1f}")
    log("  " + "-" * (len(hdr) - 2))
    tot_pairs = sum(a["pairs"] for a in agg.values())
    rec = union_found / max(total_true, 1)
    log(f"  {'UNION':<13}{rec:>10.4f}{union_found/max(tot_pairs,1):>13.2e}"
        f"{tot_pairs:>17,}{tot_pairs/max(total_s1,1):>12,.1f}")
    log(f"\n  STEP 1 RECALL    = {rec:.6f}   (of {total_true:,} true pairs)")
    log(f"  STEP 1 PRECISION = {union_found/max(tot_pairs,1):.3e}   (lower bound — "
        f"overlaps between blockers are not de-duplicated)")
    log(f"  reduction ratio  = {1 - tot_pairs/space0:.6f}  of the Step-0 space")
    log(f"  missed pairs     = {total_true - union_found:,}")
    return tot_pairs, rec


def _miss_report(ctry, union, t_sid, t_cid, s1c, candc, log):
    """STEP A — characterise the true pairs that no blocker caught."""
    miss = np.flatnonzero(~union)
    T = len(union)
    log(f"\n    STEP A — MISS ANALYSIS ({ctry}):  {len(miss):,} missed of {T:,}"
        f"  ({len(miss)/max(T,1):.2%})")
    if len(miss) == 0:
        return
    ms, mc = t_sid[miss], t_cid[miss]
    s1_nm = s1c["business_name"].to_numpy(); cd_nm = candc["business_name"].to_numpy()
    s1_core = s1c["name_core"].to_numpy();   cd_core = candc["name_core"].to_numpy()
    s1_ad = s1c["address_tokens_sorted"].to_numpy()
    cd_ad = candc["address_tokens_sorted"].to_numpy()

    nonascii = np.fromiter((not str(x).isascii() for x in cd_nm[mc]), bool, len(mc))
    empty_nm = (s1_core[ms] == "") | (cd_core[mc] == "")
    empty_ad = (s1_ad[ms] == "") | (cd_ad[mc] == "")
    log(f"      candidate name is non-Latin script ... {nonascii.sum():,} ({nonascii.mean():.1%})")
    log(f"      either cleaned name is empty ......... {empty_nm.sum():,} ({empty_nm.mean():.1%})")
    log(f"      either address is empty .............. {empty_ad.sum():,} ({empty_ad.mean():.1%})")
    log(f"      none of the above .................... {(~nonascii & ~empty_nm & ~empty_ad).sum():,}"
        f" ({(~nonascii & ~empty_nm & ~empty_ad).mean():.1%})")
    rng = np.random.default_rng(0)
    for i in rng.choice(len(miss), size=min(MISS_SAMPLE, len(miss)), replace=False):
        log(f"      S1  : {str(s1_nm[ms[i]])[:58]:<58} | {str(s1_ad[ms[i]])[:44]}")
        log(f"      CAND: {str(cd_nm[mc[i]])[:58]:<58} | {str(cd_ad[mc[i]])[:44]}")
        log("      " + "-" * 108)


# ============================================================================
#  STEP 2 — normalisation used only by the composite keys
# ============================================================================
def _L(ch):
    return ord(ch) - ord("a") + 2                # letter -> code

_IS_LET = np.zeros(39, bool); _IS_LET[2:28] = True
_IS_DIG = np.zeros(39, bool); _IS_DIG[28:38] = True
_IS_VOW = np.zeros(39, bool)
for _c in "aeiouy":
    _IS_VOW[_L(_c)] = True
_IS_V5 = np.zeros(39, bool)
for _c in "aeiou":
    _IS_V5[_L(_c)] = True
_LABIAL = np.zeros(39, bool)
for _c in "bpmvfh":                              # m before these stays m
    _LABIAL[_L(_c)] = True
_SOFT = np.zeros(39, bool)
for _c in "eiy":                                 # c / g before these are soft
    _SOFT[_L(_c)] = True
_CONF = np.arange(39, dtype=np.uint8)            # look-alike digits inside words
for _d, _c in zip("013456789", "oleasgtbg"):     # sik0ra->sikora, federa1->federal, di6ital->digital
    _CONF[28 + int(_d)] = _L(_c)
_ZERO = 28


def _spans(code):
    """token mask, token start / end positions, token index of every char."""
    t = code >= 2
    st = np.flatnonzero(t & ~np.concatenate(([False], t[:-1])))
    en = np.flatnonzero(t & ~np.concatenate((t[1:], [False]))) + 1
    mark = np.zeros(code.size, np.int64)
    mark[st] = 1
    return t, st, en, np.cumsum(mark) - 1


def _tri(code, pos):
    return (code[pos].astype(np.int64) * 1521 + code[pos + 1].astype(np.int64) * 39
            + code[pos + 2].astype(np.int64))

_T_COM, _T_NET, _T_ORG, _T_WWW = (
    sum(_L(ch) * m for ch, m in zip(w, (1521, 39, 1))) for w in ("com", "net", "org", "www"))


def _norm_name(code):
    """ 1. look-alike digits inside words     sik0ra -> sikora , 8akery -> bakery
        2. web noise                          horizoncom -> horizon , www / com dropped
        3. anusvara                           kamsaltemsi -> kansaltensi (m before a
                                              non-labial consonant is an n)          """
    code = code.copy()
    t, st, en, tid = _spans(code)
    if st.size == 0:
        return code
    has_let = np.bincount(tid[t], weights=_IS_LET[code[t]], minlength=st.size) > 0
    m = t & _IS_DIG[code]
    m[m] = has_let[tid[m]]
    code[m] = _CONF[code[m]]

    ln = en - st
    kill = np.zeros(code.size, bool)
    big = ln >= 3
    last3 = np.full(st.size, -1, np.int64); last3[big] = _tri(code, en[big] - 3)
    first3 = np.full(st.size, -1, np.int64); first3[big] = _tri(code, st[big])
    whole = (ln == 3) & np.isin(last3, [_T_COM, _T_NET, _T_ORG, _T_WWW])
    tail = (ln >= 6) & np.isin(last3, [_T_COM, _T_NET, _T_ORG])
    head = (ln >= 6) & (first3 == _T_WWW)
    kill[_expand(st[whole], ln[whole])] = True
    kill[_expand(en[tail] - 3, np.full(int(tail.sum()), 3))] = True
    kill[_expand(st[head], np.full(int(head.sum()), 3))] = True
    code = code[~kill]

    nxt = np.concatenate((code[1:], [np.uint8(0)]))
    anus = (code == _L("m")) & _IS_LET[nxt] & ~_IS_VOW[nxt] & ~_LABIAL[nxt]
    code[anus] = _L("n")
    return code



def _skel2(code, variant):
    """phonetic skeleton with consecutive-duplicate collapse.
    variant 0 = Step-1 skeleton (c->s, g->k)
    variant 1 = English-aware + transliteration-aware:
                c/g soft before e/i/y (->s), hard otherwise (->k)   creative = kriyetiva
                -tia/-tio -> s                                       foundation = favundhesan
                x -> ks                                              laxmi = laksmi
                m -> n                                               infotech = imphoteka
                n / m before a consonant dropped (nasal)             investments = investamemtsa
                                                                     royal = raonyala
                -tur- -> k (as c)                                    ventures = vemcarsa
                gh after a vowel is silent                           high care = hai keyara
    variant 2 = variant 1 with every n / m dropped (coarser, for transliterations
                that insert or lose nasals:  international = imtarnyasanal)"""
    if variant >= 1:
        code = code.copy()
        n1 = np.concatenate((code[1:], [np.uint8(0)]))
        n2 = np.concatenate((code[2:], [np.uint8(0), np.uint8(0)]))
        ts = (code == _L("t")) & (n1 == _L("i")) & ((n2 == _L("o")) | (n2 == _L("a")))
        code[ts] = _L("s")
        xm = code == _L("x")
        if xm.any():
            rep = np.where(xm, 2, 1)
            start = np.cumsum(rep) - rep
            code = np.repeat(code, rep)
            xs = start[xm]
            code[xs] = _L("k")
            code[xs + 1] = _L("s")
    s = _SKEL[code]
    if variant >= 1:
        s = s.copy()
        nxt = np.concatenate((code[1:], [np.uint8(0)]))
        nx2 = np.concatenate((code[2:], [np.uint8(0), np.uint8(0)]))
        prv = np.concatenate(([np.uint8(0)], code[:-1]))
        soft = _SOFT[nxt]
        for ch in "cg":
            m = code == _L(ch)
            s[m] = np.where(soft[m], _L("s"), _L("k"))
        tur = (code == _L("t")) & (nxt == _L("u")) & (nx2 == _L("r")) & _IS_LET[prv]
        s[tur] = _L("k")                          # -ture / -tur- sounds "ch"
        gh = (code == _L("g")) & (nxt == _L("h")) & _IS_VOW[prv]
        s[gh] = 255                               # high / light / night: silent gh
        nas = ((code == _L("n")) | (code == _L("m"))) & _IS_LET[nxt] & ~_IS_V5[nxt]
        if variant == 2:
            nas = (code == _L("n")) | (code == _L("m"))
        s[nas] = 255                              # anusvara / nasal before a consonant
        s[s == _L("m")] = _L("n")
    s = s[s != 255]
    if s.size:
        dup = np.concatenate(([False], (s[1:] == s[:-1]) & (s[1:] >= 2)))
        s = s[~dup]
    return _tidy_spaces(s)


def _norm_addr(code):
    """0064 -> 64 ,  316th / 316nd -> 316 ,  no21 -> 21"""
    t, st, en, tid = _spans(code)
    if st.size == 0:
        return code
    ln = en - st
    ndig = np.bincount(tid[t], weights=_IS_DIG[code[t]], minlength=st.size).astype(np.int64)
    kill = np.zeros(code.size, bool)

    alld = ndig == ln                                          # leading zeros
    nz = np.cumsum((code != _ZERO) & t)
    before = np.where(st > 0, nz[np.maximum(st - 1, 0)], 0)
    pos = np.arange(code.size)
    lead = t & (code == _ZERO)
    lead[lead] = alld[tid[lead]] & (nz[lead] - before[tid[lead]] == 0) & (pos[lead] < en[tid[lead]] - 1)
    kill |= lead

    big = ln >= 3                                              # ordinals
    ordn = np.zeros(st.size, bool)
    if big.any():
        a, b = code[en[big] - 2].astype(np.int64), code[en[big] - 1].astype(np.int64)
        suf = a * 39 + b
        ords = [_L(x) * 39 + _L(y) for x, y in ("st", "nd", "rd", "th")]
        ordn[big] = np.isin(suf, ords) & (ndig[big] == ln[big] - 2)
        no = np.zeros(st.size, bool)                           # "no21"
        no[big] = ((code[st[big]] == _L("n")) & (code[st[big] + 1] == _L("o"))
                   & (ndig[big] == ln[big] - 2))
        kill[_expand(st[no], np.full(int(no.sum()), 2))] = True
    kill[_expand(en[ordn] - 2, np.full(int(ordn.sum()), 2))] = True
    return code[~kill]


def _num_tokens(code, base_row):
    """all-digit address tokens of <= 8 digits -> (row, value, n_digits)."""
    t, st, en, tid = _spans(code)
    if st.size == 0:
        return np.empty(0, np.int32), np.empty(0, np.int64), np.empty(0, np.int64)
    ln = en - st
    ndig = np.bincount(tid[t], weights=_IS_DIG[code[t]], minlength=st.size)
    ok = (ndig == ln) & (ln <= 8)
    pos = np.flatnonzero(t)
    tk = tid[pos]
    keep = ok[tk]
    pos, tk = pos[keep], tk[keep]
    d = code[pos].astype(np.float64) - 28.0
    val = np.bincount(tk, weights=d * 10.0 ** (en[tk] - 1 - pos), minlength=st.size)
    rows = _rowids(code)[st].astype(np.int32) + base_row
    return rows[ok], np.rint(val[ok]).astype(np.int64), ln[ok].astype(np.int64)


def _pin_values(series):
    s = series.fillna("").astype(str).str.replace(r"\D", "", regex=True)
    ok = ((s.str.len() >= 3) & (s.str.len() <= 8)).to_numpy()
    out = np.full(len(s), -1, np.int64)
    if ok.any():
        out[ok] = s[ok].astype(np.int64).to_numpy()
    return out


_WEB = ("com", "net", "org")


def _segment(s, core, drop, memo):
    """split a glued string into the fewest dictionary words (S1 name words of >= 3
    letters, legal-form words); -> the S1 name words, or None."""
    r = memo.get(s, 0)
    if r != 0:
        return r
    L = len(s)
    INF = 1 << 30
    best = [INF] * (L + 1)
    back = [0] * (L + 1)
    best[0] = 0
    for j in range(2, L + 1):
        bj, bi = INF, -1
        for i in range(max(0, j - SEG_MAXW), j - 1):
            b = best[i]
            if b + 1 >= bj:
                continue
            w = s[i:j]
            if w in drop or (j - i >= 3 and w in core):
                bj, bi = b + 1, i
        best[j], back[j] = bj, bi
    res = None
    if best[L] < INF:
        out, j = [], L
        while j > 0:
            i = back[j]
            out.append(s[i:j])
            j = i
        keep = [w for w in reversed(out) if w not in drop]
        if keep:
            res = keep
    memo[s] = res
    return res


def _unglue(name, core, drop, memo):
    """'m s mulshiremediesprivatecom' -> 'm s mulshi remedies' (words that no S1 name
    has and that end in com / net / org are split into S1 words)."""
    toks = name.split()
    out, hit = [], False
    for t in toks:
        if len(t) >= SEG_MIN + 3 and t[-3:] in _WEB and t not in core:
            stem = t[:-3]
            if stem.startswith("www") and len(stem) > SEG_MIN + 3:
                stem = stem[3:]
            pcs = _segment(stem, core, drop, memo) if stem not in core else [stem]
            if pcs:
                out.extend(pcs)
                hit = hit or len(pcs) > 1 or pcs[0] != stem
                continue
        out.append(t)
    return (" ".join(out), True) if hit else (name, False)


def _load_s2(paths, country, flag_script=False, seg=None, want_vocab=False):
    """Step-2 loader: keeps entity ids, postal code, and name / address ALREADY
    ENCODED as byte buffers (one byte per character) instead of millions of
    Python strings — about 4x less memory than a DataFrame.
    Also keeps the full romanised name (name_translit, or business_name) so the
    legal form that preprocessing stripped from name_core can be recovered.
    want_vocab (S1): also returns the S1 name-word vocabulary and legal-form words.
    seg (candidates): that vocabulary -> glued web names are split into words."""
    ids, nm, ad, na, pins, tr, sg, al = [], [], [], [], [], [], [], []
    core, trc_counts = set(), []
    memo = {}
    base = 0
    for path in paths:
        header = pd.read_csv(ART / path, nrows=0).columns
        trc = "name_translit" if "name_translit" in header else (
            "business_name" if "business_name" in header else None)
        cols = ["entity_id", "country_clean", "name_core", "address_tokens_sorted"]
        cols += [c for c in ("address_pin", "name_primary", "name_alias") if c in header]
        cols += [c for c in (trc, "business_name" if flag_script else None) if c and c not in cols]
        for ch in pd.read_csv(ART / path, usecols=cols, dtype=str, chunksize=S2_READ):
            ch = ch[ch["country_clean"] == country]
            if len(ch) == 0:
                continue
            ch = ch.fillna("")
            names = ch["name_core"]
            trl = (ch[trc].str.lower().str.replace(r"[^a-z0-9 ]", " ", regex=True) if trc
                   else pd.Series([""] * len(ch), index=ch.index))
            if want_vocab:
                core.update(names.str.split().explode().dropna().unique().tolist())
                trc_counts.append(trl.str.split().explode().dropna().value_counts())
            flags = np.zeros(len(ch), bool)
            if seg is not None:
                m = names.str.contains("com|net|org", regex=True).to_numpy()
                if m.any():
                    fixed = [_unglue(x, seg[0], seg[1], memo) for x in names.to_numpy()[m]]
                    names = names.copy()
                    names.iloc[np.flatnonzero(m)] = [a for a, _ in fixed]
                    flags[np.flatnonzero(m)] = [b for _, b in fixed]
            ids.append(ch["entity_id"].to_numpy(dtype=object))
            nm.append((base, _encode(names.tolist())))
            ad.append((base, _encode(ch["address_tokens_sorted"].tolist())))
            tr.append((base, _encode(trl.tolist())))
            pins.append(_pin_values(ch["address_pin"]) if "address_pin" in ch
                        else np.full(len(ch), -1, np.int64))
            if "name_alias" in ch and "name_primary" in ch:
                # DBA / alias: both halves become extra exact-name keys (empty -> no key)
                al.append((base, _encode(ch["name_primary"].tolist()), _encode(ch["name_alias"].tolist())))
            sg.append(flags)
            if flag_script:
                na.append(~ch["business_name"].map(str.isascii).to_numpy(dtype=bool))
            base += len(ch)
    out = dict(n=base, ids=np.concatenate(ids) if ids else np.empty(0, object),
               name=nm, addr=ad, tr=tr, pin=np.concatenate(pins) if pins else np.empty(0, np.int64),
               nonascii=np.concatenate(na) if na else None,
               seg=np.concatenate(sg) if sg else np.zeros(0, bool), alias=al)
    if want_vocab:
        cnt = pd.concat(trc_counts).groupby(level=0).sum() if trc_counts else pd.Series(dtype=np.int64)
        legal = {w for w, c in cnt.items() if c >= SEG_LEGAL_DF and w not in core and len(w) >= 2}
        out["vocab"] = (core, legal | set(_WEB) | {"www"})
    return out


_DEC = np.array(list("\x00 abcdefghijklmnopqrstuvwxyz0123456789?"))


def _row_text(chunks, row):
    """decode one record back to text (for the miss report)."""
    bases = [b for b, _ in chunks]
    k = int(np.searchsorted(bases, row, side="right")) - 1
    b, code = chunks[k]
    ends = np.flatnonzero(code == 0)
    i = row - b
    lo = 0 if i == 0 else ends[i - 1] + 1
    return "".join(_DEC[code[lo:ends[i]]])


def _texts(chunks, rows):
    """decode many records back to text at once (miss file)."""
    rows = np.asarray(rows, np.int64)
    out = np.empty(len(rows), object)
    if len(rows) == 0:
        return out
    bases = np.array([b for b, _ in chunks], np.int64)
    k = np.searchsorted(bases, rows, side="right") - 1
    for ci in np.unique(k):
        b, code = chunks[ci]
        ends = np.flatnonzero(code == 0)
        starts = np.concatenate(([0], ends[:-1] + 1))
        for j in np.flatnonzero(k == ci):
            i = rows[j] - b
            out[j] = "".join(_DEC[code[starts[i]:ends[i]]])
    return out


_TIE_LABELS = ["1", "2-5", "6-30", "31-300", ">300"]


def _topk_matrix(rows, ids, df, n_rows, k):
    """per row, its k rarest ids (by df) as a dense (n_rows, k) int64 matrix, -1 padded."""
    M = np.full((n_rows, k), -1, dtype=np.int64)
    if len(rows) == 0 or k == 0:
        return M
    o = np.lexsort((ids, df[ids], rows))
    rows, ids = rows[o], ids[o]
    pos = np.arange(len(rows), dtype=np.int64) - _group_starts(rows, n_rows)[rows]
    m = pos < k
    M[rows[m], pos[m]] = ids[m]
    return M

# ============================================================================
#  STEP 2 — rolling hashes  (polynomial mod 2^64, concatenation-friendly:
#                            hash(a + b) = hash(a) * B^len(b) + hash(b))
# ============================================================================
_HB = 0x100000001B3                       # odd -> invertible mod 2^64
_HBI = pow(_HB, -1, 1 << 64)
_PWN = 4096


def _pow_table(b, n):
    a = np.full(n, b, dtype=np.uint64)
    a[0] = 1
    return np.cumprod(a, dtype=np.uint64)


_PW = _pow_table(_HB, _PWN)
_PWI = _pow_table(_HBI, _PWN)
_U0 = np.uint64(0)
_MIX = np.uint64(0x9E3779B97F4A7C15)
_W_SCALE = 2.0                            # largest family weight (cheap-score quantisation)


def _popcount_table(nbits):
    a = np.arange(1 << nbits, dtype=np.int64)
    c = np.zeros(a.size, np.int8)
    for b in range(nbits):
        c += ((a >> b) & 1).astype(np.int8)
    return c


_POP = _popcount_table(NF)                # evidence mask -> number of families
_MASK_ALL = (1 << NF) - 1
_PC8 = _popcount_table(8)


def _pc(x):
    """popcount of a uint64 array."""
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.int16)
    x = np.ascontiguousarray(x, dtype=np.uint64)
    return _PC8[x.view(np.uint8)].reshape(-1, 8).sum(axis=1, dtype=np.int16).reshape(x.shape)


def _tok_hash(code, base):
    """per word, in reading order: (row, H, P, length, all_digit) with
    H = sum c_i B^(len-1-i) and P = B^len."""
    t, st, en, tid = _spans(code)
    if st.size == 0:
        e = np.empty(0, np.uint64)
        return np.empty(0, np.int32), e, e, np.empty(0, np.int64), np.empty(0, bool)
    ln = (en - st).astype(np.int64)
    pos = np.flatnonzero(t)
    tk = tid[pos]
    ex = np.minimum(en[tk] - 1 - pos, _PWN - 1)
    val = code[pos].astype(np.uint64) * _PW[ex]
    seg = np.zeros(st.size, np.int64)
    seg[1:] = np.cumsum(ln)[:-1]
    H = np.add.reduceat(val, seg)
    P = _PW[np.minimum(ln, _PWN - 1)]
    nd = np.add.reduceat(_IS_DIG[code[pos]].astype(np.int64), seg)
    rows = _rowids(code)[st].astype(np.int32) + base
    return rows, H, P, ln, nd == ln


def _name_clean(code):
    """_norm_name + drop pure-number words of >= 5 digits (phone numbers, ids)."""
    code = _norm_name(code)
    t, st, en, tid = _spans(code)
    if st.size:
        ln = en - st
        nd = np.bincount(tid[t], weights=_IS_DIG[code[t]], minlength=st.size)
        bad = (nd == ln) & (ln >= 5)
        if bad.any():
            kill = np.zeros(code.size, bool)
            kill[_expand(st[bad], ln[bad])] = True
            code = code[~kill]
    return _tidy_spaces(code)


def _trunc(code, k):
    """every word cut to its first k characters (word boundaries unchanged)."""
    t, st, en, tid = _spans(code)
    if st.size == 0:
        return code
    off = np.arange(code.size, dtype=np.int64) - st[np.maximum(tid, 0)]
    return code[~t | (off < k)]


def _flat_info(code, base, del_ok=None):
    """name with spaces removed:  full hash, first-PFX_LEN-letters hash, and the
    one-typo neighbourhood (the full hash + every one-letter deletion) for
    records whose letter count is inside CD_LEN (and del_ok, if given)."""
    nrec = int(np.count_nonzero(code == 0))
    pos = np.flatnonzero(code >= 2)
    rloc = _rowids(code)[pos].astype(np.int64)
    L = np.bincount(rloc, minlength=nrec).astype(np.int64)
    rs = np.zeros(nrec, np.int64)
    rs[1:] = np.cumsum(L)[:-1]
    j = np.arange(pos.size, dtype=np.int64) - rs[rloc]
    c = code[pos].astype(np.uint64)
    q = c * _PWI[np.minimum(j, _PWN - 1)]
    cq = np.cumsum(q, dtype=np.uint64)
    has = L > 0
    before = np.zeros(nrec, np.uint64)
    m = has & (rs > 0)
    before[m] = cq[rs[m] - 1]
    tot = np.zeros(nrec, np.uint64)
    tot[has] = cq[rs[has] + L[has] - 1] - before[has]
    Lc = np.minimum(L, _PWN - 1)
    full = np.zeros(nrec, np.uint64)
    full[has] = _PW[Lc[has] - 1] * tot[has]
    pm = L > PFX_LEN
    pfx = np.zeros(nrec, np.uint64)
    pfx[pm] = _PW[PFX_LEN - 1] * (cq[rs[pm] + PFX_LEN - 1] - before[pm])

    ok = (L >= CD_LEN[0]) & (L <= CD_LEN[1])
    if del_ok is not None:
        ok &= del_ok[base:base + nrec]
    sel = ok[rloc]
    same_prev = np.zeros(pos.size, bool)
    same_prev[1:] = (c[1:] == c[:-1]) & (j[1:] > 0)      # deleting either of "ll" -> same string
    idx = np.flatnonzero(sel & ~same_prev)
    rr = rloc[idx]
    incl = cq[idx] - before[rr]
    excl = incl - q[idx]
    dk = _PW[L[rr] - 2] * excl + _PW[L[rr] - 1] * (tot[rr] - incl)
    fr = np.flatnonzero(ok)
    rows = np.concatenate([rr, fr]).astype(np.int32) + base
    keys = np.concatenate([dk, full[fr]])
    return full, pfx, rows, keys


# ---------------------------------------------------------------- signatures
# A 128-bit set signature per record (two uint64): names -> character bigrams +
# trigrams with spaces removed, addresses -> words.  |a & b| / |a | b| is a cheap
# O(1) similarity for EVERY generated pair (used to rank the pool).
_SIG_SHIFT = np.uint64(57)


def _or_bits(out, rows, g):
    """set bit hash(g) (0..127) in out[row] for every (row, g); rows non-decreasing."""
    if len(rows) == 0:
        return
    b = ((g.astype(np.uint64) + np.uint64(1)) * _MIX) >> _SIG_SHIFT
    one = np.uint64(1)
    lo = np.where(b < 64, one << (b & np.uint64(63)), _U0)
    hi = np.where(b >= 64, one << (b & np.uint64(63)), _U0)
    st = np.flatnonzero(np.concatenate(([True], rows[1:] != rows[:-1])))
    r = rows[st]
    out[r, 0] |= np.bitwise_or.reduceat(lo, st)
    out[r, 1] |= np.bitwise_or.reduceat(hi, st)


def _sig_chars(code):
    nrec = int(np.count_nonzero(code == 0))
    out = np.zeros((nrec, 2), np.uint64)
    pos = np.flatnonzero(code >= 2)
    if pos.size < 2:
        return out
    r = _rowids(code)[pos].astype(np.int64)
    c = code[pos].astype(np.int64)
    s2 = r[1:] == r[:-1]
    _or_bits(out, r[1:][s2], (c[:-1] * 64 + c[1:])[s2])
    if pos.size >= 3:
        s3 = r[2:] == r[:-2]
        _or_bits(out, r[2:][s3], (c[:-2] * 4096 + c[1:-1] * 64 + c[2:] + (1 << 20))[s3])
    return out


def _sig_pop(sg):
    return (_pc(sg[:, 0]) + _pc(sg[:, 1])).astype(np.int16)


def _seq_matrix(rows, H, P, n):
    """first W_TOK words of every record as (n, W) hash / power matrices."""
    HM = np.zeros((n, W_TOK), np.uint64)
    PM = np.ones((n, W_TOK), np.uint64)
    cnt = np.bincount(rows, minlength=n).astype(np.int64)
    if len(rows):
        k = np.arange(len(rows), dtype=np.int64) - _group_starts(rows, n)[rows]
        m = k < W_TOK
        HM[rows[m], k[m]] = H[m]
        PM[rows[m], k[m]] = P[m]
    return HM, PM, cnt


def _dedupe_rows(M):
    M = np.sort(M, axis=1)
    d = np.zeros(M.shape, bool)
    d[:, 1:] = M[:, 1:] == M[:, :-1]
    M[d] = 0
    return M


def _seq_variants(HM, PM, cnt, del2):
    """whole name, every one-word-dropped name, every two-words-dropped name.
    Words are concatenated without spaces, so 'optimal better trading' and the
    domain 'optimalbettertrading' get the same hash.  0 = no variant."""
    n, W = HM.shape
    V = [None] * (W + 1)
    S = [None] * (W + 1)
    V[W] = np.zeros(n, np.uint64)
    S[W] = np.ones(n, np.uint64)
    for j in range(W - 1, -1, -1):
        V[j] = HM[:, j] * S[j + 1] + V[j + 1]
        S[j] = PM[:, j] * S[j + 1]
    full = np.where(cnt >= 1, V[0], _U0)
    ok1 = (cnt >= 2) & (cnt <= min(W, DEL_MAX))
    D1 = np.zeros((n, W), np.uint64)
    PF = np.zeros(n, np.uint64)
    for i in range(W):
        D1[:, i] = np.where(ok1 & (i < cnt), PF * S[i + 1] + V[i + 1], _U0)
        PF = PF * PM[:, i] + HM[:, i]
    D2 = None
    if del2:
        ok2 = (cnt >= 4) & (cnt <= min(W, DEL2_MAX))
        cols = []
        PF = np.zeros(n, np.uint64)
        for i in range(W):
            g = PF.copy()
            for j in range(i + 1, W):
                cols.append(np.where(ok2 & (j < cnt), g * S[j + 1] + V[j + 1], _U0))
                g = g * PM[:, j] + HM[:, j]
            PF = PF * PM[:, i] + HM[:, i]
        D2 = _dedupe_rows(np.stack(cols, axis=1))
    return full, _dedupe_rows(D1), D2


_E_RK = (np.empty(0, np.int32), np.empty(0, np.uint64))


def _variants_subset(rows, H, P, n, sub, del2):
    """name variants of the records in `sub` only (memory) -> sparse (row, key)
    for 'full', 'D1', 'D2'."""
    idx = np.flatnonzero(sub)
    if idx.size == 0 or len(rows) == 0:
        return dict(full=_E_RK, D1=_E_RK, D2=_E_RK)
    loc = np.full(n, -1, np.int64)
    loc[idx] = np.arange(idx.size)
    m = sub[rows]
    HM, PM, cnt = _seq_matrix(loc[rows[m]], H[m], P[m], idx.size)
    full, D1, D2 = _seq_variants(HM, PM, cnt, del2)
    out = {}
    for k, M in (("full", full), ("D1", D1), ("D2", D2)):
        if M is None:
            out[k] = _E_RK
        else:
            r, v = _nz(M)
            out[k] = (idx[r].astype(np.int32), v)
    return out


def _seq_full(rows, H, P, n):
    """whole-name hash only (left fold of the word hashes)."""
    HM, PM, cnt = _seq_matrix(rows, H, P, n)
    v = np.zeros(n, np.uint64)
    for j in range(HM.shape[1]):
        v = v * PM[:, j] + HM[:, j]
    return np.where(cnt >= 1, v, _U0)


def _legal_side(tr_chunks, nm_chunks, n):
    """legal form of every record = the words preprocessing stripped from the
    name (name_translit minus name_core): an order-free hash of those words, and
    the string of their initials ('pvt ltd' = 'praiveta limiteda' -> 'pl').
    Records whose names are otherwise identical often differ here."""
    lfx = np.zeros(n, np.uint64)
    lfi = np.zeros(n, np.int64)
    for (base, tcode), (_, ncode) in zip(tr_chunks, nm_chunks):
        tr, th, tl, _, _ = _tokens_from_code(tcode, base)
        if len(tr) == 0:
            continue
        nr, nh, *_ = _tokens_from_code(ncode, base)
        core = np.sort(nh ^ (nr.astype(np.uint64) * _MIX))
        strip = ~_in_sorted(core, th ^ (tr.astype(np.uint64) * _MIX)) & (tl >= 2)
        _, st, _, _ = _spans(tcode)
        first = tcode[st].astype(np.int64)
        r, h, f = tr[strip].astype(np.int64), th[strip], first[strip]
        if len(r) == 0:
            continue
        g = np.flatnonzero(np.concatenate(([True], r[1:] != r[:-1])))
        lfx[r[g]] = np.add.reduceat((h + np.uint64(1)) * _MIX, g) | np.uint64(1)
        k = np.arange(len(r), dtype=np.int64) - np.repeat(g, np.diff(np.append(g, len(r))))
        vals = np.where(k < 4, f << (6 * np.minimum(k, 3)), 0)
        lfi[r[g]] = np.add.reduceat(vals, g)
    return lfx, lfi


def _lf_pair(a, b):
    """1 same legal form, -1 different, -0.5 only one side has one, 0 neither."""
    ha, hb = a != 0, b != 0
    return np.where(ha & hb, np.where(a == b, 1.0, -1.0),
                    np.where(ha ^ hb, -0.5, 0.0)).astype(np.float32)


def _idf(df, n):
    """scale-free rarity weight in [0, 1]; 0 for a word no S1 record has."""
    w = (np.log((n + 1.0) / (df + 1.0)) / np.log(n + 1.0)).astype(np.float32)
    w[df == 0] = 0.0
    return w


def _ids1(M):
    """int64 id matrix (-1 = none) -> uint64 (id + 1, 0 = none)."""
    return np.where(M >= 0, (M + 1).astype(np.uint64), _U0)


def _nz(M):
    if M.ndim == 1:
        r = np.flatnonzero(M)
        return r.astype(np.int32), M[r]
    r, k = np.nonzero(M)
    return r.astype(np.int32), M[r, k]


def _rk(parts, rows_ok=None):
    """dense key matrices / vectors and sparse (row, key) pairs -> one (row, key)."""
    R, K = [], []
    for M in parts:
        if M is None:
            continue
        r, k = M if isinstance(M, tuple) else _nz(M)
        if rows_ok is not None and len(r):
            m = rows_ok[r]
            r, k = r[m], k[m]
        R.append(r.astype(np.int32)); K.append(k)
    if not R:
        return _E_RK
    return np.concatenate(R), np.concatenate(K)


def _part(r, k, part, parts):
    if parts > 1:
        m = (k % np.uint64(parts)) == np.uint64(part)
        return r[m], k[m]
    return r, k


def _cross(Xs, Y, part, parts):
    """every (name variant x address word) of a record; Xs = dense matrices or
    sparse (row, key) pairs."""
    R, K = [], []

    def emit(ra, xv):
        Yr = Y[ra]
        xv = xv * _MIX
        for b in range(Y.shape[1]):
            m = Yr[:, b] != 0
            r, k = _part(ra[m].astype(np.int32), xv[m] + Yr[m, b], part, parts)
            R.append(r); K.append(k)

    for X in Xs:
        if isinstance(X, tuple):
            if len(X[0]):
                emit(X[0].astype(np.int64), X[1])
            continue
        for a in range(X.shape[1]):
            ra = np.flatnonzero(X[:, a])
            if ra.size:
                emit(ra, X[ra, a])
    if not R:
        return _E_RK
    return np.concatenate(R), np.concatenate(K)


def _within(M, part, parts):
    R, K = [], []
    k = M.shape[1]
    for a in range(k):
        for b in range(a + 1, k):
            x, y = M[:, a], M[:, b]
            r = np.flatnonzero((x != 0) & (y != 0))
            lo, hi = np.minimum(x[r], y[r]), np.maximum(x[r], y[r])
            rr, kk = _part(r.astype(np.int32), (lo << np.uint64(32)) | hi, part, parts)
            R.append(rr); K.append(kk)
    if not R:
        return _E_RK
    return np.concatenate(R), np.concatenate(K)


def _csr(rows, vals, n):
    """rows sorted -> offsets (n+1)."""
    return _group_starts(rows.astype(np.int64), n)


# ============================================================================
#  STEP 2a — record representation (both sides)
# ============================================================================
_X = np.uint8(_L("x"))


def _addr_side(chunks, s1_side=False):
    """address words (+ all-digit flag), signature, number tokens and phonetic words.
    S1 side: every number of >= 3 digits also yields its masked form 'xx' + rest
    (1297 -> xx97), the way some sources write it ('##97' -> preprocessing 'xx97')."""
    tr, th, tf, nr, nv, nl, sg, kr, kh = [], [], [], [], [], [], [], [], []
    for base, raw in chunks:
        code = _norm_addr(raw)
        r, h, ln, _, ad = _tokens_from_code(code, base)
        s = np.zeros((int(np.count_nonzero(code == 0)), 2), np.uint64)
        _or_bits(s, (r - base).astype(np.int64), h)
        sg.append(s)
        if s1_side and len(r):
            m = ad & (ln >= 3)
            if m.any():
                _, st, _, _ = _spans(code)
                c2 = code.copy()
                c2[st[m]] = _X
                c2[st[m] + 1] = _X
                r2, h2, *_ = _tokens_from_code(c2, base)
                r, h = np.concatenate([r, r2[m]]), np.concatenate([h, h2[m]])
                ad = np.concatenate([ad, np.zeros(int(m.sum()), bool)])
        tr.append(r); th.append(h); tf.append(ad)
        r, v, l = _num_tokens(code, base)
        nr.append(r); nv.append(v); nl.append(l)
        sk = _skel2(code, 1)
        r, h, ln, hd, _ = _tokens_from_code(sk, base)
        m = ~hd & (ln >= SKA_MIN)
        kr.append(r[m]); kh.append(h[m])
    cat = lambda x, dt: np.concatenate(x) if x else np.empty(0, dt)
    return (cat(tr, np.int32), cat(th, np.uint64), cat(tf, bool), cat(nr, np.int32),
            cat(nv, np.int64), cat(nl, np.int64),
            np.concatenate(sg) if sg else np.zeros((0, 2), np.uint64),
            cat(kr, np.int32), cat(kh, np.uint64))


_NS_DT = dict(rr=np.int32, kr=np.int32, qr=np.int32, pr=np.int32, cr=np.int32,
              rl=np.int64, kl=np.int64, ql=np.int64)


def _name_side(chunks, del_ok):
    keys = ("rr", "rh", "rp", "rl", "kr", "kh", "kp", "kl", "qr", "qh", "qp", "ql",
            "pr", "ph", "pp", "full", "pfx", "cr", "ck", "sgN", "sgK",
            "t1h", "t1p", "t2h", "t2p")
    o = {k: [] for k in keys}
    for base, raw in chunks:
        code = _name_clean(raw)
        r, h, p, l, _ = _tok_hash(code, base)
        o["rr"].append(r); o["rh"].append(h); o["rp"].append(p); o["rl"].append(l)
        sk = _skel2(code, 1)
        r, h, p, l, _ = _tok_hash(sk, base)
        o["kr"].append(r); o["kh"].append(h); o["kp"].append(p); o["kl"].append(l)
        r, h, p, l, _ = _tok_hash(_skel2(code, 2), base)
        o["qr"].append(r); o["qh"].append(h); o["qp"].append(p); o["ql"].append(l)
        r, h, p, _, _ = _tok_hash(_trunc(code, PFX3), base)
        o["pr"].append(r); o["ph"].append(h); o["pp"].append(p)
        # first letter / first two letters of every word (aligned with the raw words)
        _, h, p, _, _ = _tok_hash(_trunc(code, 1), base)
        o["t1h"].append(h); o["t1p"].append(p)
        _, h, p, _, _ = _tok_hash(_trunc(code, 2), base)
        o["t2h"].append(h); o["t2p"].append(p)
        full, pfx, cr, ck = _flat_info(code, base, del_ok)
        o["full"].append(full); o["pfx"].append(pfx); o["cr"].append(cr); o["ck"].append(ck)
        o["sgN"].append(_sig_chars(code)); o["sgK"].append(_sig_chars(sk))
    out = {}
    for k, v in o.items():
        if k in ("sgN", "sgK"):
            out[k] = np.concatenate(v) if v else np.zeros((0, 2), np.uint64)
        else:
            out[k] = np.concatenate(v) if v else np.empty(0, _NS_DT.get(k, np.uint64))
    return out


def _full_addr_id(r, k, n):
    """order-free hash of the (normalised, unique) address word set; 0 = none."""
    out = np.zeros(n, np.uint64)
    if len(r) == 0:
        return out
    h = (k.astype(np.uint64) + np.uint64(1)) * _MIX
    g = np.flatnonzero(np.concatenate(([True], r[1:] != r[:-1])))
    out[r[g]] = np.add.reduceat(h, g) | np.uint64(1)
    return out


def _vocab_side(sr, sh, cr, ch, n1, n2, flags=None):
    """shared word vocabulary for one kind of word -> per side (row, id) unique,
    document frequencies and IDF weights; c_tok = id of every candidate word in
    reading order.  flags = (S1, candidate) per-word booleans -> isnum[id]."""
    sk, ck = _compact(sh, ch)
    V = int(max(sk.max(initial=-1), ck.max(initial=-1))) + 1
    isnum = None
    if flags is not None:
        isnum = np.zeros(max(V, 1), bool)
        isnum[sk[flags[0]]] = True
        isnum[ck[flags[1]]] = True
    s_r, s_k = _dedupe(sr, sk)
    c_r, c_k = _dedupe(cr, ck)
    df_s = np.bincount(s_k, minlength=V)
    df_c = np.bincount(c_k, minlength=V)
    return dict(V=max(V, 1), s_r=s_r, s_k=s_k, c_r=c_r, c_k=c_k, df_s=df_s, df_c=df_c,
                w=_idf(df_s, n1), c_tok=ck, isnum=isnum)


def _noise_keep(Vd, rows, lens, n1, n2):
    """candidate words to keep in the name keys.  Dropped: short words that are
    frequent among candidate names but (nearly) absent from S1 names — source
    noise such as a transliterated legal form ('pra li') or a prefix ('m s').
    Data-driven, so the same rule serves every country.  A name is never emptied."""
    ctok = Vd["c_tok"]
    if len(ctok) == 0:
        return np.ones(0, bool), 0, 0.0
    Lw = np.zeros(Vd["V"], np.int64)
    Lw[ctok] = lens
    df_s, df_c = Vd["df_s"].astype(np.float64), Vd["df_c"].astype(np.float64)
    noise = ((Lw > 0) & (Lw <= NOISE_LEN) & (df_c >= NOISE_DF)
             & (df_c * n1 >= NOISE_RATIO * (df_s + 1.0) * n2))
    nt = noise[ctok]
    has_clean = np.bincount(rows[~nt], minlength=n2) > 0
    keep = ~nt | ~has_clean[rows]
    hit = np.bincount(rows[~keep], minlength=n2) > 0
    return keep, int(noise.sum()), float(hit.mean()) if n2 else 0.0


_SET_MIX = np.uint64(0xD6E8FEB86659FD93)


def _set_keys(HM, cnt, side_s):
    """order-free word-set hash of the first W_TOK words: the whole set, and (S1 side)
    the set minus 1 word (names of >= 3 words) / minus 2 words (>= 4 words)."""
    mx = np.where(HM != 0, (HM + np.uint64(1)) * _SET_MIX, _U0)
    tot = mx.sum(axis=1, dtype=np.uint64)
    full = np.where(cnt >= 1, tot | np.uint64(1), _U0)
    if not side_s:
        return full, _E_RK
    n, W = HM.shape
    cols = []
    for i in range(W):
        cols.append(np.where((cnt >= 3) & (i < cnt), (tot - mx[:, i]) | np.uint64(1), _U0))
    for i in range(W):
        for j in range(i + 1, W):
            cols.append(np.where((cnt >= 4) & (j < cnt), (tot - mx[:, i] - mx[:, j]) | np.uint64(1), _U0))
    D = _dedupe_rows(np.stack(cols, axis=1))
    return full, _nz(D)


def _numx(nr, nv, nl, n):
    """house numbers of >= 3 digits without their last digit -> sparse (row, key)."""
    m = nl >= 3
    if not m.any():
        return _E_RK
    k = ((nv[m] // 10).astype(np.uint64) + np.uint64(7)) * _SET_MIX
    r = nr[m].astype(np.int64)
    o = np.lexsort((k, r))
    r, k = r[o], k[o]
    d = np.concatenate(([True], (r[1:] != r[:-1]) | (k[1:] != k[:-1])))
    return r[d].astype(np.int32), k[d]


_NUM_MIX = np.uint64(0xA24BAED4963EE407)
_FWI1_X = np.uint64(0x5851F42D4C957F2D)
_FWI2_X = np.uint64(0x14057B7EF767814F)
_SK1_X = np.uint64(0x2545F4914F6CDD1D)


def _num_mats(sv, cv, n1, n2, k_s, k_c):
    """ad_numw number part.  sv / cv = (rows, values, n_digits) of the all-digit address
    tokens.  Per record its k rarest numbers (<= 6 digits, rarity among S1) -> dense
    (n, k) uint64 keys, and every pair of them -> dense (n, k(k-1)/2) keys.  0 = none."""
    def prep(r, v, l):
        m = l <= 6
        return r[m].astype(np.int64), v[m]
    sr, sv_ = prep(*sv)
    cr, cv_ = prep(*cv)
    uv = np.unique(np.concatenate([sv_, cv_]))
    if uv.size == 0:
        uv = np.zeros(1, np.int64)
    si = np.searchsorted(uv, sv_)
    ci = np.searchsorted(uv, cv_)
    sr, si = _dedupe(sr, si)
    cr, ci = _dedupe(cr, ci)
    df = np.bincount(si, minlength=max(len(uv), 1))
    out = []
    for r, i, n, k in ((sr, si, n1, k_s), (cr, ci, n2, k_c)):
        M = _topk_matrix(r, i, df, n, k)
        H = np.where(M >= 0, (uv[np.maximum(M, 0)].astype(np.uint64) + np.uint64(11)) * _NUM_MIX, _U0)
        pairs = []
        for a in range(k):
            for b in range(a + 1, k):
                x, y = H[:, a], H[:, b]
                lo, hi = np.minimum(x, y), np.maximum(x, y)
                pairs.append(np.where((x != 0) & (y != 0), (lo * _MIX + hi) | np.uint64(1), _U0))
        P = np.stack(pairs, axis=1) if pairs else np.zeros((n, 0), np.uint64)
        out.append((H, P))
    return out


def _fwi(rows, Hr, Pr, Ht, Pt, n, ntok, lens=None, minlen=3):
    """first word in full + the first ntok words after it truncated (Ht / Pt = the
    truncated word hashes, aligned with the raw ones).  0 for one-word names and
    (lens given) for names whose first word is shorter than minlen ('m s ...', 'x ...')."""
    HM, PM, cnt = _seq_matrix(rows, Hr, Pr, n)
    HT, PT, _ = _seq_matrix(rows, Ht, Pt, n)
    v = HM[:, 0].copy()
    for j in range(1, min(ntok + 1, HM.shape[1])):
        v = v * PT[:, j] + HT[:, j]
    ok = cnt >= 2
    if lens is not None and len(rows):
        first = np.concatenate(([True], rows[1:] != rows[:-1]))
        fl = np.zeros(n, np.int64)
        fl[rows[first]] = lens[first]
        ok &= fl >= minlen
    return np.where(ok, v, _U0)


def _alias_keys(al, n, full):
    """DBA / alias halves (name_primary, name_alias) -> sparse (row, raw-name hash),
    skipping empty ones and ones equal to the whole name."""
    R, K = [], []
    for base, pcode, acode in al:
        for code in (pcode, acode):
            nrec = int(np.count_nonzero(code == 0))
            r, h, p, _, _ = _tok_hash(_name_clean(code), 0)
            v = _seq_full(r, h, p, nrec)
            rows = np.flatnonzero(v != 0)
            g = rows + base
            keep = v[rows] != full[g]
            R.append(g[keep].astype(np.int32)); K.append(v[rows][keep])
    if not R:
        return _E_RK
    return np.concatenate(R), np.concatenate(K)


def _rep(S1, CD, log):
    """-> XS, XC (what each side contributes to keys) and B (what the pair
    features need)."""
    t0 = time.time()
    n1, n2 = S1["n"], CD["n"]
    B = dict(n1=n1, n2=n2)

    # independent heavy pieces run in parallel (numpy releases the GIL)
    ex = cf.ThreadPoolExecutor(max_workers=3)
    f_as = ex.submit(_addr_side, S1["addr"], True)
    f_ac = ex.submit(_addr_side, CD["addr"], False)
    f_ns = ex.submit(_name_side, S1["name"], None)
    f_ls = ex.submit(_legal_side, S1["tr"], S1["name"], n1)
    f_lc = ex.submit(_legal_side, CD["tr"], CD["name"], n2)

    # ------------------------------------------------------------ addresses
    s_ar, s_ah, s_af, s_nr, s_nv, s_nl, B["sgA_s"], s_kr, s_kh = f_as.result()
    c_ar, c_ah, c_af, c_nr, c_nv, c_nl, B["sgA_c"], c_kr, c_kh = f_ac.result()
    B["pcA_s"], B["pcA_c"] = _sig_pop(B["sgA_s"]), _sig_pop(B["sgA_c"])
    A = _vocab_side(s_ar, s_ah, c_ar, c_ah, n1, n2, flags=(s_af, c_af))
    del s_ar, s_ah, c_ar, c_ah, s_af, c_af
    live_c = A["df_s"][A["c_k"]] > 0
    weak = np.bincount(A["c_r"][live_c], minlength=n2) == 0      # no address word any S1 has
    B["weak"] = weak
    adf_s = _full_addr_id(A["s_r"], A["s_k"], n1)
    adf_c = _full_addr_id(A["c_r"], A["c_k"], n2)
    B["adf_s"], B["adf_c"] = adf_s, adf_c
    u, inv, cnt = np.unique(adf_s, return_inverse=True, return_counts=True)
    B["addr_amb"] = np.where(adf_s != 0, cnt[inv.ravel()], 0).astype(np.float32)

    ks = max(K_AP[0], K_FA[0], K_NA[0][1], K_NX[0], K_FZ[0][0])
    kc = max(K_AP[1], K_FA[1], K_NA[1][1], K_NX[1], K_FZ[1][0])
    ms = A["df_c"][A["s_k"]] > 0
    ADs = _ids1(_topk_matrix(A["s_r"][ms], A["s_k"][ms], A["df_s"], n1, ks))
    ADc = _ids1(_topk_matrix(A["c_r"][live_c], A["c_k"][live_c], A["df_s"], n2, kc))
    # v7: the rarest NON-number words (ad_numw) ...
    ma = ms & ~A["isnum"][A["s_k"]]
    la = live_c & ~A["isnum"][A["c_k"]]
    ADa_s = _ids1(_topk_matrix(A["s_r"][ma], A["s_k"][ma], A["df_s"], n1, K_NW[0][1]))
    ADa_c = _ids1(_topk_matrix(A["c_r"][la], A["c_k"][la], A["df_s"], n2, K_NW[1][1]))
    del ma, la
    # ... and the rarest phonetic address words (fz_addr, ad_skpair)
    SV = _vocab_side(s_kr, s_kh, c_kr, c_kh, n1, n2)
    del s_kr, s_kh, c_kr, c_kh
    m1 = SV["df_c"][SV["s_k"]] > 0
    m2 = SV["df_s"][SV["c_k"]] > 0
    SKA_s = _ids1(_topk_matrix(SV["s_r"][m1], SV["s_k"][m1], SV["df_s"], n1, max(K_FZ[0][1], K_SP[0])))
    SKA_c = _ids1(_topk_matrix(SV["c_r"][m2], SV["c_k"][m2], SV["df_s"], n2, max(K_FZ[1][1], K_SP[1])))
    del SV, m1, m2
    (NV_s, NP_s), (NV_c, NP_c) = _num_mats((s_nr, s_nv, s_nl), (c_nr, c_nv, c_nl), n1, n2,
                                           K_NW[0][0], K_NW[1][0])

    B["A_SL"] = A["s_r"].astype(np.int64) * A["V"] + A["s_k"]
    B["A_V"] = A["V"]
    B["A_off"] = _csr(A["c_r"], A["c_k"], n2)
    B["A_ids"] = A["c_k"]
    B["A_w"] = A["w"][A["c_k"]]
    B["A_sum_s"] = np.bincount(A["s_r"], weights=A["w"][A["s_k"]], minlength=n1).astype(np.float32)
    B["A_sum_c"] = np.bincount(A["c_r"], weights=B["A_w"], minlength=n2).astype(np.float32)
    B["A_n_s"] = np.bincount(A["s_r"], minlength=n1).astype(np.float32)
    B["A_n_c"] = np.bincount(A["c_r"][live_c], minlength=n2).astype(np.float32)
    del A, ms, live_c
    low = weak | (B["A_sum_c"] < LOW_ADDR_W)        # the address cannot tell S1 records apart
    B["low"] = low
    f_nc = ex.submit(_name_side, CD["name"], low)   # typo neighbourhood only for low-address candidates

    # house numbers: exact values and S1-side noise variants (2300 ~ 300, 837 ~ 83)
    NX_s, NX_c = _numx(s_nr, s_nv, s_nl, n1), _numx(c_nr, c_nv, c_nl, n2)
    SC = np.int64(1_000_000_000)
    sx = s_nr.astype(np.int64) * SC + s_nv
    var = [sx]
    for cond, vv in ((s_nl >= 4, s_nv % 1000), (s_nl >= 3, s_nv % 100), (s_nl >= 3, s_nv // 10)):
        var.append(s_nr[cond].astype(np.int64) * SC + vv[cond])
    B["N_SX"] = np.unique(sx)
    B["N_SV"] = np.unique(np.concatenate(var))
    cn = np.unique(c_nr.astype(np.int64) * SC + c_nv)
    B["N_off"] = _csr((cn // SC), None, n2)
    B["N_ids"] = cn % SC
    B["N_n_c"] = np.diff(B["N_off"]).astype(np.float32)
    del s_nr, s_nv, s_nl, c_nr, c_nv, c_nl, sx, var, cn
    B["pin_s"], B["pin_c"] = S1["pin"], CD["pin"]
    B["seg_c"] = CD["seg"]

    # ------------------------------------------------------------ legal form
    B["lfx_s"], B["lfi_s"] = f_ls.result()
    B["lfx_c"], B["lfi_c"] = f_lc.result()

    # ------------------------------------------------------------ names
    sn = f_ns.result()
    cn = f_nc.result()
    ex.shutdown()
    for tg in ("N", "K"):
        B[f"sg{tg}_s"], B[f"sg{tg}_c"] = sn[f"sg{tg}"], cn[f"sg{tg}"]
        B[f"pc{tg}_s"], B[f"pc{tg}_c"] = _sig_pop(sn[f"sg{tg}"]), _sig_pop(cn[f"sg{tg}"])
    NR = _vocab_side(sn["rr"], sn["rh"], cn["rr"], cn["rh"], n1, n2)
    NK = _vocab_side(sn["kr"], sn["kh"], cn["kr"], cn["kh"], n1, n2)
    for tag, Vd in (("R", NR), ("K", NK)):
        B[f"{tag}_SL"] = Vd["s_r"].astype(np.int64) * Vd["V"] + Vd["s_k"]
        B[f"{tag}_V"] = Vd["V"]
        B[f"{tag}_off"] = _csr(Vd["c_r"], Vd["c_k"], n2)
        B[f"{tag}_ids"] = Vd["c_k"]
        B[f"{tag}_w"] = Vd["w"][Vd["c_k"]]
        B[f"{tag}_sum_s"] = np.bincount(Vd["s_r"], weights=Vd["w"][Vd["s_k"]], minlength=n1).astype(np.float32)
        B[f"{tag}_sum_c"] = np.bincount(Vd["c_r"], weights=B[f"{tag}_w"], minlength=n2).astype(np.float32)
        B[f"{tag}_n_s"] = np.bincount(Vd["s_r"], minlength=n1).astype(np.float32)
        B[f"{tag}_n_c"] = np.bincount(Vd["c_r"], minlength=n2).astype(np.float32)

    # candidate-only noise words leave the candidate's name keys
    keepR, nzR, hitR = _noise_keep(NR, cn["rr"], cn["rl"], n1, n2)
    keepK, nzK, _ = _noise_keep(NK, cn["kr"], cn["kl"], n1, n2)
    NQ = _vocab_side(sn["qr"], sn["qh"], cn["qr"], cn["qh"], n1, n2)
    keepQ, _, _ = _noise_keep(NQ, cn["qr"], cn["ql"], n1, n2)
    del NQ

    XS, XC = dict(n=n1), dict(n=n2, low=low)
    XS["s2_full"] = _seq_full(sn["qr"], sn["qh"], sn["qp"], n1)
    XC["s2_full"] = _seq_full(cn["qr"][keepQ], cn["qh"][keepQ], cn["qp"][keepQ], n2)
    del keepQ
    seqs = {"s": (("raw", sn["rr"], sn["rh"], sn["rp"]), ("sk", sn["kr"], sn["kh"], sn["kp"])),
            "c": (("raw", cn["rr"][keepR], cn["rh"][keepR], cn["rp"][keepR]),
                  ("sk", cn["kr"][keepK], cn["kh"][keepK], cn["kp"][keepK]))}
    for side, X in (("s", XS), ("c", XC)):
        for tag, r, h, p in seqs[side]:
            HM, PM, cnt = _seq_matrix(r, h, p, X["n"])
            full, D1, D2 = _seq_variants(HM, PM, cnt, side == "s")
            if tag == "raw":
                X["set_full"], X["set_D"] = _set_keys(HM, cnt, side == "s")
            del HM, PM
            X[f"{tag}_full"], X[f"{tag}_D1"] = full, D1
            if side == "s":
                X[f"{tag}_D2"] = _nz(D2)
            else:                              # candidates: only 4..5-word names have them
                sub = (cnt >= 4) & (cnt <= min(W_TOK, DEL2_MAX))
                X[f"{tag}_D2"] = _variants_subset(r, h, p, X["n"], sub, True)["D2"]
            del D2
            if tag == "raw":
                X["rsmall"] = cnt <= FA_MAX
            if tag == "sk":
                small = cnt <= FA_MAX
                s2 = np.where(X["s2_full"] != full, X["s2_full"], _U0)    # no duplicate keys
                X["FA"] = np.concatenate([full[:, None], s2[:, None],
                                          np.where(small[:, None], D1, _U0)], axis=1)
                del s2
                if side == "c":
                    d2r, d2k = X["sk_D2"]
                    m = small[d2r]
                    X["FA2"] = (d2r[m], d2k[m])
            del cnt
        X["pfx"] = (sn if side == "s" else cn)["pfx"]
        d = sn if side == "s" else cn
        X["cd_r"], X["cd_k"] = d["cr"], d["ck"]
    del seqs

    # first letters of every word (typos rarely hit them): S1 all, candidates low-address only
    HM, PM, cnt = _seq_matrix(sn["pr"], sn["ph"], sn["pp"], n1)
    f3, d13, _ = _seq_variants(HM, PM, cnt, False)
    # v7: first word + initials / + first two letters of the next FWI_N words
    xo = lambda v, c: np.where(v != 0, v ^ c, _U0)
    fw_s = [xo(_fwi(sn["rr"], sn["rh"], sn["rp"], sn[f"t{k}h"], sn[f"t{k}p"], n1, FWI_N), c)
            for k, c in ((1, _FWI1_X), (2, _FWI2_X))]
    XS["p3"] = _rk([f3, d13] + fw_s)
    del HM, PM, cnt, f3, d13, fw_s
    v = _variants_subset(cn["pr"][keepR], cn["ph"][keepR], cn["pp"][keepR], n2, low, False)
    cr_, crh, crp = cn["rr"][keepR], cn["rh"][keepR], cn["rp"][keepR]
    fw_c = [xo(_fwi(cr_, crh, crp, cn[f"t{k}h"][keepR], cn[f"t{k}p"][keepR], n2, FWI_N), c)
            for k, c in ((1, _FWI1_X), (2, _FWI2_X))]
    fr, fk = _rk(fw_c, low)                          # candidates: low-address only
    vr, vk = _rk([v["full"], v["D1"]])
    XC["p3"] = (np.concatenate([vr, fr]), np.concatenate([vk, fk]))
    del v, fw_c, fr, fk, vr, vk

    # v7 fz_addr name parts: first word + first letter of the second (typo-proof) ...
    XS["PC2"] = _fwi(sn["rr"], sn["rh"], sn["rp"], sn["t1h"], sn["t1p"], n1, 1, sn["rl"])[:, None]
    XC["PC2"] = _fwi(cr_, crh, crp, cn["t1h"][keepR], cn["t1p"][keepR], n2, 1, cn["rl"][keepR])[:, None]
    del cr_, crh, crp

    # ... and the phonetic first word (>= 2 letters)
    def _sk1(r, h, ln, n):
        out = np.zeros(n, np.uint64)
        if len(r):
            first = np.concatenate(([True], r[1:] != r[:-1])) & (ln >= 2)
            out[r[first]] = h[first] ^ _SK1_X
        return out[:, None]
    XS["SK1"] = _sk1(sn["kr"], sn["kh"], sn["kl"], n1)
    XC["SK1"] = _sk1(cn["kr"][keepK], cn["kh"][keepK], cn["kl"][keepK], n2)
    del keepR, keepK

    # candidate's first name word: how many S1 names contain it (report: tie size)
    fwdf = np.zeros(n2, np.float32)
    if len(cn["rr"]):
        first = np.flatnonzero(np.concatenate(([True], cn["rr"][1:] != cn["rr"][:-1])))
        fwdf[cn["rr"][first]] = NR["df_s"][NR["c_tok"][first]]
    B["fwdf_c"] = fwdf

    # DBA / alias halves as whole-name keys
    XS["AL"] = _alias_keys(S1.get("alias", []), n1, XS["raw_full"])
    XC["AL"] = _alias_keys(CD.get("alias", []), n2, XC["raw_full"])

    B["raw_s"], B["raw_c"] = XS["raw_full"], XC["raw_full"]
    B["sk_s"], B["sk_c"] = XS["sk_full"], XC["sk_full"]
    B["s2_s"], B["s2_c"] = XS["s2_full"], XC["s2_full"]
    B["set_s"], B["set_c"] = XS["set_full"], XC["set_full"]
    B["pfx_s"], B["pfx_c"] = XS["pfx"], XC["pfx"]
    u, inv, cnt = np.unique(XS["raw_full"], return_inverse=True, return_counts=True)
    B["name_amb"] = np.where(XS["raw_full"] != 0, cnt[inv.ravel()], 0).astype(np.float32)
    kl = XS["raw_full"] ^ (B["lfi_s"].astype(np.uint64) * _MIX)    # same name AND legal form
    u, inv, cnt = np.unique(kl, return_inverse=True, return_counts=True)
    B["name_lf_amb"] = np.where(XS["raw_full"] != 0, cnt[inv.ravel()], 0).astype(np.float32)
    B["lamb"] = np.log1p(B["name_amb"]).astype(np.float32)
    del sn, cn, u, inv, cnt, kl

    # single rare name words: S1 = all its words, candidate = its rarest
    XS["tok_r"] = np.concatenate([NR["s_r"], NK["s_r"]])
    XS["tok_k"] = np.concatenate([((NR["s_k"] + 1).astype(np.uint64) << np.uint64(1)),
                                  ((NK["s_k"] + 1).astype(np.uint64) << np.uint64(1)) | np.uint64(1)])
    mr = NR["df_s"][NR["c_k"]] > 0
    mk = NK["df_s"][NK["c_k"]] > 0
    TR = _ids1(_topk_matrix(NR["c_r"][mr], NR["c_k"][mr], NR["df_s"], n2, K_TOK[0]))
    kcn = max(K_NA[1][0], K_TOK[1])
    XC["NK"] = _ids1(_topk_matrix(NK["c_r"][mk], NK["c_k"][mk], NK["df_s"], n2, kcn))
    r1, k1 = _nz(TR)
    r2, k2 = _nz(XC["NK"][:, :K_TOK[1]])
    XC["tok_r"] = np.concatenate([r1, r2])
    XC["tok_k"] = np.concatenate([k1 << np.uint64(1), (k2 << np.uint64(1)) | np.uint64(1)])
    ms = NK["df_c"][NK["s_k"]] > 0
    XS["NK"] = _ids1(_topk_matrix(NK["s_r"][ms], NK["s_k"][ms], NK["df_s"], n1, K_NA[0][0]))
    XS["AD"], XC["AD"] = ADs, ADc
    XS["NX"], XC["NX"] = NX_s, NX_c
    XS["ad_full"], XC["ad_full"] = adf_s, adf_c
    XS["ADa"], XC["ADa"] = ADa_s, ADa_c
    XS["SKA"], XC["SKA"] = SKA_s, SKA_c
    XS["NV"], XS["NP"], XC["NV"], XC["NP"] = NV_s, NP_s, NV_c, NP_c
    del NR, NK, TR, mr, mk, ms
    gc.collect()
    lf_c, lf_s = float((B["lfi_c"] != 0).mean()), float((B["lfi_s"] != 0).mean())
    n_al = len(XS["AL"][0]) + len(XC["AL"][0])
    log(f"    2a  record keys ................ {n1:,} S1 / {n2:,} candidates  "
        f"({weak.mean():.1%} of candidates have no usable address, {low.mean():.1%} a low / no address;  "
        f"{nzR + nzK:,} candidate-only noise words dropped from {hitR:.1%} of candidate names;  "
        f"{int(B['seg_c'].sum()):,} glued web names split;  "
        f"legal form known for {lf_s:.1%} of S1 / {lf_c:.1%} of candidates;  "
        f"{n_al:,} alias name keys)"
        f"   [{time.time()-t0:.1f}s]")
    return XS, XC, B


def _fam_keys(f, side, X, part, parts):
    """(rows, keys) that family f draws from one side."""
    s = side == "s"
    i = 0 if s else 1
    if f[:4] in ("nmF_", "skF_"):
        pre = "raw" if f[0] == "n" else "sk"
        tag = f[4:]
        full, D1, D2 = X[f"{pre}_full"], X[f"{pre}_D1"], X[f"{pre}_D2"]
        if tag == "FF":
            mats = [full] + ([X["AL"]] if pre == "raw" else [])      # + DBA / alias halves
        elif tag == "SF":
            mats = [D1, D2] if s else [full]
        elif tag == "F1":
            mats = [full] if s else [D1, D2]
        else:
            mats = [D1]
        return _part(*_rk(mats), part, parts)
    if f in ("lw_name", "lw_skel"):
        pre = "raw" if f == "lw_name" else "sk"
        mats = [X[f"{pre}_full"], X[f"{pre}_D1"], X[f"{pre}_D2"]] + ([X["AL"]] if pre == "raw" else [])
        return _part(*_rk(mats, None if s else X["low"]), part, parts)
    if f == "fz_addr":
        a = _cross([X["PC2"]], X["AD"][:, :K_FZ[i][0]], part, parts)
        b = _cross([X["SK1"]], X["SKA"][:, :K_FZ[i][1]], part, parts)
        return np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1]])
    if f == "ad_numw":
        return _cross([X["NV"], X["NP"]], X["ADa"][:, :K_NW[i][1]], part, parts)
    if f == "ad_skpair":
        return _within(X["SKA"][:, :K_SP[i]], part, parts)
    if f == "lw_pfx3":
        return _part(*X["p3"], part, parts)          # candidate side: low-address only
    if f == "s2F_FF":
        return _part(*_nz(X["s2_full"]), part, parts)
    if f == "nm_set":
        return _part(*_rk([X["set_full"], X["set_D"]] if s else [X["set_full"]]), part, parts)
    if f == "ad_numx":
        return _cross([X["NX"]], X["AD"][:, :K_NX[i]], part, parts)
    if f == "rw_addr":
        RA = [X["raw_full"][:, None], np.where(X["rsmall"][:, None], X["raw_D1"], _U0)]
        return _cross(RA, X["AD"][:, :K_FA[i]], part, parts)
    if f == "nm_pfx":
        return _part(*_nz(X["pfx"]), part, parts)
    if f == "nm_char":
        return _part(X["cd_r"], X["cd_k"], part, parts)
    if f == "nm_tok":
        return _part(X["tok_r"], X["tok_k"], part, parts)
    if f == "nm_addr":
        return _cross([X["NK"][:, :K_NA[i][0]]], X["AD"][:, :K_NA[i][1]], part, parts)
    if f == "fl_addr":
        return _cross([X["FA"]] + ([X["FA2"]] if "FA2" in X else []), X["AD"][:, :K_FA[i]], part, parts)
    if f == "ad_full":
        return _part(*_nz(X["ad_full"]), part, parts)
    if f == "ad_pair":
        return _within(X["AD"][:, :K_AP[i]], part, parts)
    raise KeyError(f)


# ============================================================================
#  STEP 2b — key index
# ============================================================================
def _index_task(f, part, XS, XC):
    """count one family (one key-space part): keys both sides share and under the caps."""
    cap_s, cap_c, _, parts = FAMILIES[f]
    sr, sk = _fam_keys(f, "s", XS, part, parts)
    cr, ck = _fam_keys(f, "c", XC, part, parts)
    if len(sk) == 0 or len(ck) == 0:
        return None
    # sort-based counting; the inverse maps every posting to its key
    # without random lookups (fast at 100M+ postings)
    us, sinv, cs = np.unique(sk, return_inverse=True, return_counts=True)
    uc, cinv, cc = np.unique(ck, return_inverse=True, return_counts=True)
    del sk, ck
    p = np.searchsorted(uc, us)
    pc = np.minimum(p, len(uc) - 1)
    both = (p < len(uc)) & (uc[pc] == us)
    cc_s = np.where(both, cc[pc], 0)
    # candidate cap grows when few S1 share the key (few pairs per candidate)
    lim = np.maximum(cap_c, np.minimum(cap_c * CC_BOOST, (cap_c * cap_s) // np.maximum(cs, 1)))
    ok = both & (cs <= cap_s) & (cc_s <= lim)
    nk = int(ok.sum())
    new_s = np.where(ok, np.cumsum(ok) - 1, -1)          # S1 unique key -> kept id
    q = np.searchsorted(us, uc)
    qc = np.minimum(q, len(us) - 1)
    new_c = np.where((q < len(us)) & (us[qc] == uc), new_s[qc], -1)
    out = []
    for rows, inv, new in ((sr, sinv, new_s), (cr, cinv, new_c)):
        kid = new[inv.ravel()]
        m = kid >= 0
        out.append((rows[m], kid[m].astype(np.int32)))
    return dict(s=out[0], c=out[1], kn1=cs[ok].astype(np.int32), kn2=cc_s[ok].astype(np.int32), nk=nk,
                pairs=int((cs[ok].astype(np.int64) * cc_s[ok]).sum()))


def _build_index(XS, XC, n1, n2, log):
    """count every key of every family, keep keys both sides share and under
    the family caps, pack as CSR:  key -> S1 rows,  candidate -> keys."""
    t0 = time.time()
    SR, SK, CR, CK, KN1, KN2, KF = [], [], [], [], [], [], []
    base = 0
    stats = {f: [0, 0] for f in FAM}
    tasks = [(fi, f, part) for fi, f in enumerate(FAM) for part in range(FAMILIES[f][3])]
    with cf.ThreadPoolExecutor(max_workers=max(1, IDX_THREADS)) as ex:
        for (fi, f, part), r in zip(tasks, ex.map(lambda t: _index_task(t[1], t[2], XS, XC), tasks)):
            if r is None:
                continue
            nk = r["nk"]
            SR.append(r["s"][0]); SK.append(r["s"][1] + np.int32(base))
            CR.append(r["c"][0]); CK.append(r["c"][1] + np.int32(base))
            KN1.append(r["kn1"]); KN2.append(r["kn2"])
            KF.append(np.full(nk, fi, np.int8))
            stats[f][0] += nk
            stats[f][1] += r["pairs"]
            base += nk
    gc.collect()

    kn1 = np.concatenate(KN1) if KN1 else np.empty(0, np.int32)
    kn2 = np.concatenate(KN2) if KN2 else np.empty(0, np.int32)
    fam = np.concatenate(KF) if KF else np.empty(0, np.int8)
    s_rows = np.concatenate(SR) if SR else np.empty(0, np.int32)
    s_kid = np.concatenate(SK) if SK else np.empty(0, np.int32)
    c_rows = np.concatenate(CR) if CR else np.empty(0, np.int32)
    c_kid = np.concatenate(CK) if CK else np.empty(0, np.int32)
    del SR, SK, CR, CK

    # ---- runtime guard: drop the most expensive keys if over budget ----
    cost = kn1.astype(np.int64) * kn2
    total = int(cost.sum())
    dropped = 0
    if total > S2_MAX_POST:
        o = np.argsort(cost)
        cut = np.searchsorted(np.cumsum(cost[o]), S2_MAX_POST, side="right")
        keep = np.zeros(len(kn1), bool); keep[o[:cut]] = True
        dropped = int((~keep).sum())
        remap = (np.cumsum(keep) - 1).astype(np.int32)
        m = keep[s_kid]; s_rows, s_kid = s_rows[m], remap[s_kid[m]]
        m = keep[c_kid]; c_rows, c_kid = c_rows[m], remap[c_kid[m]]
        kn1, kn2, fam = kn1[keep], kn2[keep], fam[keep]
        total = int((kn1.astype(np.int64) * kn2).sum())
    nk = len(kn1)

    o = np.argsort(s_kid, kind="stable")
    key_s1 = s_rows[o]
    key_off = _group_starts(s_kid[o].astype(np.int64), nk)
    del o, s_rows, s_kid
    o = np.argsort(c_rows, kind="stable")
    cand_key = c_kid[o]
    cand_off = _group_starts(c_rows[o].astype(np.int64), n2)
    del o, c_rows, c_kid

    # evidence of a shared key = how surprising it is that a random S1 shares it
    # (scale-free: 1.0 for a key unique to one S1), times the family weight
    fwt = np.array([FAMILIES[f][2] for f in FAM], np.float32)
    # (a key far more common among candidates than S1 counts suggest is weaker evidence)
    eff = np.maximum(kn1.astype(np.float64), kn2 * (n1 / max(n2, 1)))
    kw = np.maximum(1.0 - np.log(np.maximum(eff, 1.0)) / np.log(max(n1, 3)), 0.0).astype(np.float32) * fwt[fam]
    wq = np.clip(np.rint(kw / _W_SCALE * 65535), 1, 65535).astype(np.uint64)
    fwp = (fam.astype(np.uint64) << np.uint64(16)) | wq
    log(f"    2b  key index .................. {nk:,} keys, {total:,} pair postings "
        f"({total/max(n1,1):,.0f}/S1)" + (f", guard dropped {dropped:,} keys" if dropped else "")
        + f"   [{time.time()-t0:.1f}s]")
    log("        " + "  ".join(f"{f} {stats[f][0]:,}" for f in FAM))
    return dict(key_s1=key_s1, key_off=key_off, cand_key=cand_key, cand_off=cand_off,
                n1key=np.diff(key_off).astype(np.int32), fw=fwp, postings=total, fstats=stats)


# ============================================================================
#  STEP 2c — pairs of one candidate batch
# ============================================================================
def _gen_batch(ix, lo, hi, n1):
    """every (candidate, S1) pair of candidates [lo, hi): cheap score (sum of key
    evidence), number of shared keys, family bitmask.  One sort of packed
    uint64 (pair id | family | weight) replaces unique + bincounts."""
    co = ix["cand_off"]
    k_lo, k_hi = int(co[lo]), int(co[hi])
    if k_hi == k_lo:
        return None
    kids = ix["cand_key"][k_lo:k_hi]
    per = np.diff(co[lo:hi + 1])
    cnt = ix["n1key"][kids].astype(np.int64)
    if int(cnt.sum()) == 0:
        return None
    cand = np.repeat(np.repeat(np.arange(hi - lo, dtype=np.uint64), per), cnt)
    s1 = ix["key_s1"][_expand(ix["key_off"][kids], cnt)].astype(np.uint64)
    v = ((cand * np.uint64(n1) + s1) << np.uint64(21)) | np.repeat(ix["fw"][kids], cnt)
    del cand, s1
    v.sort()
    pid = v >> np.uint64(21)
    st = np.flatnonzero(np.concatenate(([True], pid[1:] != pid[:-1])))
    uniq = pid[st].astype(np.int64)
    del pid
    score = np.add.reduceat((v & np.uint64(0xFFFF)).astype(np.float32), st) * np.float32(_W_SCALE / 65535)
    fam = ((v >> np.uint64(16)) & np.uint64(31)).astype(np.int32)
    mask = np.bitwise_or.reduceat(np.int32(1) << fam, st).astype(np.int32)
    nkeys = np.diff(np.append(st, len(v))).astype(np.int32)
    return uniq, score.astype(np.float32), nkeys, mask


def _rank_desc(group, val):
    """rank of each element inside its group by descending val (any sign), and
    the group best."""
    u = np.ascontiguousarray(val, dtype=np.float32).view(np.uint32).astype(np.uint64)
    neg = (u >> np.uint64(31)) == 1
    u = np.where(neg, np.uint64(0xFFFFFFFF) - u, u | np.uint64(0x80000000))   # monotone in val
    key = (np.asarray(group).astype(np.uint64) << np.uint64(32)) | (np.uint64(0xFFFFFFFF) - u)
    del u, neg
    o = np.argsort(key)
    del key
    g = np.asarray(group)[o]
    first = np.concatenate(([True], g[1:] != g[:-1]))
    start = np.maximum.accumulate(np.where(first, np.arange(len(g)), 0))
    rank = np.empty(len(g), np.int64)
    rank[o] = np.arange(len(g)) - start
    best = np.empty(len(g), np.float32)
    best[o] = np.asarray(val, np.float32)[o][start]
    return rank, best


def _groups(c):
    """start and length of each run of equal values in a grouped array."""
    gst = np.flatnonzero(np.concatenate(([True], c[1:] != c[:-1])))
    return gst, np.diff(np.append(gst, len(c)))


def _grp_max(v, gst, gl):
    return np.repeat(np.maximum.reduceat(v, gst), gl)


def _fam_counts(mask):
    """(2, NF): pairs carrying family i, and pairs carrying ONLY family i."""
    out = np.zeros((2, NF), np.int64)
    if len(mask) == 0:
        return out
    single = (mask & (mask - 1)) == 0
    for i in range(NF):
        b = (mask & (1 << i)) != 0
        out[0, i] = int(np.count_nonzero(b))
        out[1, i] = int(np.count_nonzero(b & single))
    return out


def _sig_inter(B, tag, s, c):
    a, b = B[f"sg{tag}_s"], B[f"sg{tag}_c"]
    inter = (_pc(a[s, 0] & b[c, 0]) + _pc(a[s, 1] & b[c, 1])).astype(np.float32)
    return inter, B[f"pc{tag}_s"][s].astype(np.float32), B[f"pc{tag}_c"][c].astype(np.float32)


# ============================================================================
#  STEP 2d — learned pre-score (ranks EVERY generated pair; picks the pool)
# ============================================================================
PRE_FEATS = (["score", "lnk", "rel"] + [f"{a}{t}" for t in "NKA" for a in ("j", "j2_", "g", "cc", "cs")]
             + ["lfx", "lfi", "eq_raw", "eq_sk", "eq_s2", "eq_set", "eq_addr", "pin",
                "low_jN", "low_jK", "low_rel", "low_eq", "lamb", "low_lamb"])


def _eq(a, b):
    return ((a != 0) & (a == b)).astype(np.float32)


def _pre_cols(B, c, s, gst, gl, score, nkeys, best):
    """generated-pair features, one column at a time (order = PRE_FEATS).
    N = name characters, K = phonetic name characters, A = address words;
    j = Jaccard, g = gap to the candidate's best j, cc / cs = share of the
    candidate / S1 signature covered."""
    yield score
    yield np.log1p(nkeys).astype(np.float32)
    yield (score / np.maximum(best, 1e-6)).astype(np.float32)
    js = {}
    for tag in "NKA":
        inter, pa, pb = _sig_inter(B, tag, s, c)
        j = inter / np.maximum(pa + pb - inter, 1.0)
        if tag != "A":
            js[tag] = j
        yield j
        yield j * j
        yield j - _grp_max(j, gst, gl)
        yield inter / np.maximum(pb, 1.0)
        yield inter / np.maximum(pa, 1.0)
    yield _lf_pair(B["lfx_s"][s], B["lfx_c"][c])
    yield _lf_pair(B["lfi_s"][s], B["lfi_c"][c])
    eqr = _eq(B["raw_c"][c], B["raw_s"][s])
    yield eqr
    yield _eq(B["sk_c"][c], B["sk_s"][s])
    yield _eq(B["s2_c"][c], B["s2_s"][s])
    yield _eq(B["set_c"][c], B["set_s"][s])
    yield _eq(B["adf_c"][c], B["adf_s"][s])
    pc, ps = B["pin_c"][c], B["pin_s"][s]
    yield np.where((pc >= 0) & (ps >= 0), np.where(pc == ps, 1.0, -1.0), 0.0).astype(np.float32)
    lw = B["low"][c].astype(np.float32)
    yield lw * js.pop("N")
    yield lw * js.pop("K")
    yield lw * (score / np.maximum(best, 1e-6)).astype(np.float32)
    yield lw * eqr
    la = B["lamb"][s]
    yield la
    yield lw * la


def _pre_default():
    """hand weights (no fitted pre-score available)."""
    w = np.zeros(len(PRE_FEATS), np.float32)
    for k, v in (("score", 1.0), ("rel", 1.0), ("jN", 2.0), ("jK", 1.0), ("jA", 1.5),
                 ("eq_raw", 2.0), ("eq_sk", 1.0), ("eq_set", 1.0), ("eq_addr", 1.0)):
        w[PRE_FEATS.index(k)] = v
    return dict(w=w, wf=np.zeros(NF, np.float32), wpop=0.1, b=0.0, feats=list(PRE_FEATS))


def _pre_table(pre):
    """evidence mask -> intercept + family terms (one lookup per pair)."""
    a = np.arange(1 << NF, dtype=np.int64)
    T = np.full(a.size, pre["b"], np.float64) + pre["wpop"] * _POP
    for i in range(NF):
        T += pre["wf"][i] * ((a >> i) & 1)
    return T.astype(np.float32)


def _pre_score(pre, T, cols, mask):
    z = T[mask].astype(np.float32)
    for w, col in zip(pre["w"], cols):
        if w != 0:
            z += np.float32(w) * col
    return z


def _fit_pre(X, mask, y):
    """logistic regression on generated pairs -> linear weights on the raw scale."""
    if len(y) == 0 or y.min() == y.max():
        return None
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None
    bits = ((mask[:, None].astype(np.int64) >> np.arange(NF)) & 1).astype(np.float32)
    Z = np.concatenate([X, bits, _POP[mask].astype(np.float32)[:, None]], axis=1)
    del bits
    mu = Z.mean(0)
    sd = Z.std(0)
    sd[sd < 1e-6] = 1.0
    Z -= mu
    Z /= sd
    m = LogisticRegression(C=1.0, max_iter=300)
    m.fit(Z, y)
    del Z
    coef = m.coef_[0] / sd
    b = float(m.intercept_[0] - (coef * mu).sum())
    k = X.shape[1]
    return dict(w=coef[:k].astype(np.float32), wf=coef[k:k + NF].astype(np.float32),
                wpop=float(coef[k + NF]), b=b, feats=list(PRE_FEATS))


# ============================================================================
#  STEP 2e — pair features on the pool
# ============================================================================
FEAT_NAMES = (["score", "log_nkeys", "n_fam", "rank_c", "rel_c", "log_nopt", "weak", "low", "ad_sum_c"]
              + [f"has_{f}" for f in FAM]
              + ["nr_w", "nr_cc", "nr_cs", "nk_w", "nk_cc", "nk_cs", "nr_nc", "nr_ns", "nk_n",
                 "eq_raw", "eq_sk", "eq_pfx", "eq_s2", "eq_set", "seg_c", "name_amb", "name_lf_amb",
                 "lfx", "lfi", "lf_c", "lf_s",
                 "ad_w", "ad_cc", "ad_cs", "ad_n", "ad_nc", "ad_ns", "eq_addr", "addr_amb",
                 "num_c", "num_ex", "num_var", "pin",
                 "jN", "ccN", "csN", "jK", "ccK", "csK", "jA", "ccA", "csA",
                 "gap_nk_cc", "gap_nr_w", "gap_ad_cc", "gap_ad_w", "gap_jN", "gap_jK", "gap_jA"])
_FI = {f: i for i, f in enumerate(FEAT_NAMES)}


def _inter(c, s, off, ids, w, SL, V):
    """weighted and plain count of candidate words found in the S1 record.
    Pairs arrive sorted by S1, so the lookups are near-sequential."""
    n = len(c)
    cnt = (off[c + 1] - off[c]).astype(np.int64)
    if n == 0 or int(cnt.sum()) == 0:
        return np.zeros(n, np.float32), np.zeros(n, np.float32)
    pi = np.repeat(np.arange(n, dtype=np.int64), cnt)
    pos = _expand(off[c], cnt)
    q = s[pi] * np.int64(V) + ids[pos]
    hit = _in_sorted(SL, q)
    ph = pi[hit]
    iw = np.bincount(ph, weights=(w[pos[hit]] if w is not None else None), minlength=n)
    ic = np.bincount(ph, minlength=n) if w is not None else iw
    return iw.astype(np.float32), ic.astype(np.float32)


def _stage_b(c, s, ctx, B):
    """feature matrix for pool pairs (c = candidate row, s = S1 row, both int64).
    Rows must be grouped by candidate (they are: pool order is candidate-major)."""
    n = len(c)
    X = np.zeros((n, len(FEAT_NAMES)), np.float32)
    if n == 0:
        return X
    X[:, _FI["score"]] = ctx["score"]
    X[:, _FI["log_nkeys"]] = np.log1p(ctx["nkeys"])
    X[:, _FI["n_fam"]] = _POP[ctx["mask"]]
    X[:, _FI["rank_c"]] = np.minimum(ctx["prank"], 30)
    X[:, _FI["rel_c"]] = ctx["score"] / np.maximum(ctx["best"], 1e-6)
    X[:, _FI["log_nopt"]] = np.log1p(ctx["nopt"])
    X[:, _FI["weak"]] = B["weak"][c]
    X[:, _FI["low"]] = B["low"][c]
    X[:, _FI["ad_sum_c"]] = B["A_sum_c"][c]
    for i, f in enumerate(FAM):
        X[:, _FI[f"has_{f}"]] = (ctx["mask"] >> i) & 1

    o = np.argsort(s, kind="stable")                 # near-sequential S1 lookups
    cs_, ss_ = c[o], s[o]
    res = {}
    for tag in ("R", "K", "A"):
        iw, ic = _inter(cs_, ss_, B[f"{tag}_off"], B[f"{tag}_ids"], B[f"{tag}_w"], B[f"{tag}_SL"], B[f"{tag}_V"])
        res[tag] = (iw, ic)
    nx, _ = _inter(cs_, ss_, B["N_off"], B["N_ids"], None, B["N_SX"], 1_000_000_000)
    nv, _ = _inter(cs_, ss_, B["N_off"], B["N_ids"], None, B["N_SV"], 1_000_000_000)
    back = np.empty(n, np.int64)
    back[o] = np.arange(n)
    un = lambda a: a[back]

    eps = np.float32(1e-6)
    rw = un(res["R"][0])
    kw_, kc_ = un(res["K"][0]), un(res["K"][1])
    aw, ac = un(res["A"][0]), un(res["A"][1])
    X[:, _FI["nr_w"]] = rw
    X[:, _FI["nr_cc"]] = rw / np.maximum(B["R_sum_c"][c], eps)
    X[:, _FI["nr_cs"]] = rw / np.maximum(B["R_sum_s"][s], eps)
    X[:, _FI["nk_w"]] = kw_
    X[:, _FI["nk_cc"]] = kw_ / np.maximum(B["K_sum_c"][c], eps)
    X[:, _FI["nk_cs"]] = kw_ / np.maximum(B["K_sum_s"][s], eps)
    X[:, _FI["nr_nc"]] = B["R_n_c"][c]
    X[:, _FI["nr_ns"]] = B["R_n_s"][s]
    X[:, _FI["nk_n"]] = kc_
    X[:, _FI["eq_raw"]] = (B["raw_c"][c] != 0) & (B["raw_c"][c] == B["raw_s"][s])
    X[:, _FI["eq_sk"]] = (B["sk_c"][c] != 0) & (B["sk_c"][c] == B["sk_s"][s])
    X[:, _FI["eq_pfx"]] = (B["pfx_c"][c] != 0) & (B["pfx_c"][c] == B["pfx_s"][s])
    X[:, _FI["eq_s2"]] = (B["s2_c"][c] != 0) & (B["s2_c"][c] == B["s2_s"][s])
    X[:, _FI["eq_set"]] = _eq(B["set_c"][c], B["set_s"][s])
    X[:, _FI["seg_c"]] = B["seg_c"][c]
    X[:, _FI["name_amb"]] = np.log1p(B["name_amb"][s])
    X[:, _FI["name_lf_amb"]] = np.log1p(B["name_lf_amb"][s])
    X[:, _FI["lfx"]] = _lf_pair(B["lfx_s"][s], B["lfx_c"][c])
    X[:, _FI["lfi"]] = _lf_pair(B["lfi_s"][s], B["lfi_c"][c])
    X[:, _FI["lf_c"]] = B["lfi_c"][c] != 0
    X[:, _FI["lf_s"]] = B["lfi_s"][s] != 0
    X[:, _FI["ad_w"]] = aw
    X[:, _FI["ad_cc"]] = aw / np.maximum(B["A_sum_c"][c], eps)
    X[:, _FI["ad_cs"]] = aw / np.maximum(B["A_sum_s"][s], eps)
    X[:, _FI["ad_n"]] = ac
    X[:, _FI["ad_nc"]] = B["A_n_c"][c]
    X[:, _FI["ad_ns"]] = B["A_n_s"][s]
    X[:, _FI["eq_addr"]] = (B["adf_c"][c] != 0) & (B["adf_c"][c] == B["adf_s"][s])
    X[:, _FI["addr_amb"]] = np.log1p(B["addr_amb"][s])
    X[:, _FI["num_c"]] = B["N_n_c"][c]
    X[:, _FI["num_ex"]] = un(nx)
    X[:, _FI["num_var"]] = un(nv)
    pc, ps = B["pin_c"][c], B["pin_s"][s]
    X[:, _FI["pin"]] = np.where((pc >= 0) & (ps >= 0), np.where(pc == ps, 1, -1), 0)
    for tag in "NKA":
        inter, pa, pb = _sig_inter(B, tag, s, c)
        X[:, _FI[f"j{tag}"]] = inter / np.maximum(pa + pb - inter, 1.0)
        X[:, _FI[f"cc{tag}"]] = inter / np.maximum(pb, 1.0)
        X[:, _FI[f"cs{tag}"]] = inter / np.maximum(pa, 1.0)

    # the same features relative to the candidate's best option in the pool
    gst, gl = _groups(c)
    for src in ("nk_cc", "nr_w", "ad_cc", "ad_w", "jN", "jK", "jA"):
        v = X[:, _FI[src]]
        X[:, _FI["gap_" + src]] = v - _grp_max(v, gst, gl)
    return X


# ============================================================================
#  STEP 2f — blocking ranker
# ============================================================================
def _fit_ranker(X, y):
    try:
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=63, max_bin=63,
                               min_child_samples=100, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, n_jobs=-1, verbose=-1, random_state=0)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=63,
                                           min_samples_leaf=100, random_state=0)
    m.fit(X, y)
    return m


def _predict(model, X):
    return model.predict_proba(X)[:, 1].astype(np.float32)


def _fallback_p(X):
    """no ranker available (test without a train run): a monotone blend of the
    strongest features, in [0, 1]."""
    return (0.3 * X[:, _FI["rel_c"]] + 0.2 * np.clip(X[:, _FI["nk_cc"]], 0, 1)
            + 0.2 * np.clip(X[:, _FI["ad_cc"]], 0, 1) + 0.3 * X[:, _FI["jN"]]).astype(np.float32)


def _auc(y, p):
    y = y.astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(p).rank().to_numpy()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ============================================================================
#  STEP 2g — context re-score: every shortlisted pair seen next to its competitors
# ============================================================================
CTX_FEATS = ["lp", "rank_c", "n_c", "lp_best_c", "lp_2nd_c", "gap_c", "share_c", "sum_c", "near_c",
             "rank_s", "n_s", "lp_best_s", "gap_s", "share_s", "sum_s", "other_s", "nbest_s",
             "low", "weak", "lamb", "eq_raw"]


def _logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).astype(np.float32)


def _ctx_feats(c, s, p, B):
    """c, s = candidate / S1 rows of the shortlist, p = ranker probability."""
    n = len(p)
    X = np.zeros((n, len(CTX_FEATS)), np.float32)
    if n == 0:
        return X
    ci, si = c.astype(np.int64), s.astype(np.int64)
    n1, n2 = B["n1"], B["n2"]
    p = p.astype(np.float32)
    lp = _logit(p)
    rc, bc = _rank_desc(ci, p)
    rs, bs = _rank_desc(si, p)
    nc = np.bincount(ci, minlength=n2).astype(np.float32)
    ns = np.bincount(si, minlength=n1).astype(np.float32)
    sc = np.bincount(ci, weights=p, minlength=n2).astype(np.float32)
    ss = np.bincount(si, weights=p, minlength=n1).astype(np.float32)
    sec = np.full(n2, 1e-6, np.float32)
    m = rc == 1
    sec[ci[m]] = p[m]
    near = np.bincount(ci, weights=(p >= 0.5 * bc), minlength=n2).astype(np.float32)
    nb = np.bincount(si, weights=(rc == 0), minlength=n1).astype(np.float32)
    F = {"lp": lp, "rank_c": np.minimum(rc, 50), "n_c": nc[ci], "lp_best_c": _logit(bc),
         "lp_2nd_c": _logit(sec[ci]), "gap_c": lp - _logit(bc), "share_c": p / np.maximum(sc[ci], 1e-6),
         "sum_c": sc[ci], "near_c": near[ci],
         "rank_s": np.minimum(rs, 50), "n_s": ns[si], "lp_best_s": _logit(bs), "gap_s": lp - _logit(bs),
         "share_s": p / np.maximum(ss[si], 1e-6), "sum_s": ss[si], "other_s": ss[si] - p, "nbest_s": nb[si],
         "low": B["low"][ci], "weak": B["weak"][ci], "lamb": B["lamb"][si],
         "eq_raw": _eq(B["raw_c"][ci], B["raw_s"][si])}
    for i, f in enumerate(CTX_FEATS):
        X[:, i] = F[f]
    return X


def _fit_stack(X, y):
    try:
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.08, num_leaves=31, max_bin=127,
                               min_child_samples=200, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.9, n_jobs=-1, verbose=-1, random_state=0)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=31,
                                           min_samples_leaf=200, random_state=0)
    m.fit(X, y)
    return m


def _auto_budget(p, lab, n1, T):
    """pair budget (pairs per S1) chosen for recall:
       the smallest budget in AUTO_GRID whose recall reaches
           min(TARGET_RECALL, shortlist recall - RECALL_SLACK)
       unless its precision is below MIN_PRECISION -> then the largest budget that
       keeps MIN_PRECISION.   -> (budget, how it was chosen)"""
    if len(p) == 0 or T == 0:
        return DEFAULT_PER_S1, "default"
    o = np.argsort(-p, kind="stable")
    ct = np.cumsum(lab[o].astype(np.int64))
    need = min(TARGET_RECALL * T, ct[-1] - RECALL_SLACK * T)
    grid = [b for b in AUTO_GRID if b <= MAX_PER_S1]
    floor_b = None                                 # largest budget with precision >= MIN_PRECISION
    for b in grid:
        k = min(int(b * n1), len(o))
        if k == 0:
            continue
        prec = ct[k - 1] / k
        if prec >= MIN_PRECISION:
            floor_b = b
        if ct[k - 1] >= need:
            if prec >= MIN_PRECISION:
                return float(b), f"auto: recall target {need / T:.4f} reached"
            return float(floor_b if floor_b is not None else grid[0]), \
                f"auto: precision floor {MIN_PRECISION} (recall target {need / T:.4f} needs {b:.1f})"
    if floor_b is None:
        return float(grid[0]), "auto: precision floor not reachable"
    return float(floor_b), f"auto: recall target {need / T:.4f} not reached within {MAX_PER_S1}"


def _load_ranker(log):
    """-> (ranker model or None, pre-score weights, context model or None, budget or None)"""
    if not RANKER_PATH.exists():
        log(f"  NOTE: {RANKER_PATH} not found -> fallback scores "
            f"(run `python blocking.py` on train first)")
        return None, _pre_default(), None, None
    import joblib
    b = joblib.load(RANKER_PATH)
    if (b.get("features") != FEAT_NAMES or b.get("fams") != FAM
            or b.get("pre_feats") != PRE_FEATS or b.get("pre") is None
            or b.get("ctx_feats") != CTX_FEATS):
        log("  NOTE: saved blocking ranker was built by a different blocking.py version -> "
            "fallback scores (re-run `python blocking.py` on train)")
        return None, _pre_default(), None, None
    log(f"  blocking ranker loaded from {RANKER_PATH.name}  (trained on {b.get('n_train', 0):,} pairs"
        f"; pair budget {b.get('budget')})")
    return b["model"], b["pre"], b.get("stack"), b.get("budget")


# ============================================================================
#  STEP 2 — one country
# ============================================================================
_STRONG_BITS = sum(1 << FAM.index(f) for f in STRONG)


def _safe(s):
    return "".join(ch if ch.isalnum() else "_" for ch in str(s))


def _budget_pick(p, budget):
    """indices of the `budget` highest-p pairs (all of them if fewer)."""
    if budget >= len(p):
        return np.arange(len(p))
    if budget <= 0:
        return np.empty(0, np.int64)
    k = np.argpartition(-p, budget - 1)[:budget]
    return np.sort(k)


def _budget_curve(p, lab, n1, grid):
    """(pairs, true pairs, p threshold) if the budget were b x n1 pairs, for b in grid."""
    o = np.argsort(-p, kind="stable")
    ct = np.cumsum(lab[o].astype(np.int64)) if len(o) else np.zeros(0, np.int64)
    out = {}
    for b in grid:
        k = min(int(b * n1), len(o))
        out[b] = (k, int(ct[k - 1]) if k else 0, float(p[o[k - 1]]) if k else 1.0)
    return out


def _country(split, ctry, truth, bundle, log):
    """generate, score and select the candidate pairs of one country.
    Writes its rows to a part file and returns counters for the report."""
    tc0 = time.time()
    cfg = FILES[split]
    has_truth = truth is not None
    model, pre, stack, budget_saved = bundle if bundle is not None else (None, _pre_default(), None, None)
    S1 = _load_s2([cfg["s1"]], ctry, want_vocab=True)
    CD = _load_s2(cfg["cd"], ctry, flag_script=has_truth, seg=S1.pop("vocab"))
    n1, n2 = S1["n"], CD["n"]
    R = dict(ctry=ctry, n1=n1, n2=n2, T=0, part=None, pooled=None, pre_pooled=None,
             stk_pooled=None, tax={}, samples=[], tie={}, miss_rows=[])
    log(f"\n  {ctry}:  S1={n1:,}  candidates={n2:,}   [loaded {time.time()-tc0:.1f}s]")
    if n1 == 0 or n2 == 0:
        return R

    tk_all = None
    if has_truth:
        ts = pd.Index(S1["ids"]).get_indexer(truth["s1"])
        tc = pd.Index(CD["ids"]).get_indexer(truth["cand"])
        ok = (ts >= 0) & (tc >= 0)
        tk_all = np.unique(tc[ok].astype(np.int64) * n1 + ts[ok])
        R["T"] = len(tk_all)
        R["multi"] = int((np.bincount(tc[ok], minlength=n2) > 1).sum())
        del ts, tc, ok

    XS, XC, B = _rep(S1, CD, log)
    s1_ids, cd_ids = S1["ids"], CD["ids"]
    nonascii = CD["nonascii"]
    want_txt = has_truth and (MISS_SAMPLE > 0 or MISS_CSV)
    txt = (S1, CD) if want_txt else None
    del S1, CD
    ix = _build_index(XS, XC, n1, n2, log)
    del XS, XC
    gc.collect()
    R["postings"] = ix["postings"]
    R["fstats"] = ix["fstats"]
    low = B["low"]

    cs = np.zeros(len(ix["cand_key"]) + 1, np.int64)
    cs[1:] = np.cumsum(ix["n1key"][ix["cand_key"]])
    cost = cs[ix["cand_off"]]
    del cs
    bounds = [0]
    while bounds[-1] < n2:
        nxt = int(np.searchsorted(cost, cost[bounds[-1]] + S2_BATCH, side="right")) - 1
        nxt = min(nxt, bounds[-1] + 1_000_000)          # keeps packed pair ids < 2^43
        bounds.append(min(n2, max(nxt, bounds[-1] + 1)))
    batches = list(zip(bounds[:-1], bounds[1:]))
    del cost

    def gen_of(lo, hi):
        g = _gen_batch(ix, lo, hi, n1)
        if g is None:
            return None
        uniq, score, nkeys, mask = g
        cl = uniq // n1
        gst, gl = _groups(cl)
        return dict(uniq=uniq, score=score, nkeys=nkeys, mask=mask, cl=cl, c=cl + lo,
                    s=uniq % n1, gst=gst, gl=gl, best=_grp_max(score, gst, gl))

    def cols_of(G):
        return _pre_cols(B, G["c"], G["s"], G["gst"], G["gl"], G["score"], G["nkeys"], G["best"])

    def labels(ids_global):
        return _in_sorted(tk_all, ids_global)

    rng = np.random.default_rng(0)

    # ---- 2d-i  fit the pre-score (train) ----
    if has_truth:
        t0 = time.time()
        samp = batches[::PRE_EVERY]
        per = max(PRE_SAMPLE // max(len(samp), 1), 1)
        XL, ML, YL = [], [], []
        for lo, hi in samp:
            G = gen_of(lo, hi)
            if G is None:
                continue
            y = labels(G["c"] * n1 + G["s"])
            pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
            npos = min(len(pos), per // 3)
            k = np.concatenate([rng.choice(pos, size=npos, replace=False) if npos < len(pos) else pos,
                                rng.choice(neg, size=min(len(neg), per - npos), replace=False)])
            k.sort()
            XL.append(np.stack([col[k] for col in cols_of(G)], axis=1).astype(np.float32))
            ML.append(G["mask"][k]); YL.append(y[k].astype(np.int8))
            del G, y, pos, neg, k
        if YL:
            X = np.concatenate(XL); M = np.concatenate(ML); Y = np.concatenate(YL)
            del XL, ML, YL
            fitted = _fit_pre(X, M, Y)
            pre = fitted if fitted is not None else _pre_default()
            T = _pre_table(pre)
            z = T[M] + X @ pre["w"]
            k = rng.choice(len(Y), size=min(PRE_POOLED, len(Y)), replace=False)
            R["pre_pooled"] = (X[k], M[k], Y[k])
            log(f"    2d  pre-score: {len(Y):,} generated pairs (positives {Y.mean():.3f}), "
                f"AUC {_auc(Y, z):.4f}{'' if fitted else '  (fallback weights)'}   [{time.time()-t0:.1f}s]")
            del X, M, Y, z, k
    T_pre = _pre_table(pre)

    def pool_of(lo, hi, sub_rows=None):
        G = gen_of(lo, hi)
        if G is None:
            return None
        z = _pre_score(pre, T_pre, cols_of(G), G["mask"])
        prank, _ = _rank_desc(G["cl"], z)
        del z
        strong = (G["mask"] & _STRONG_BITS) != 0
        pm = np.flatnonzero((prank < np.where(low[G["c"]], POOL_K_LOW, POOL_K))
                            | (strong & (prank < POOL_STRONG)))
        if sub_rows is not None and len(pm) > sub_rows:
            # ranker sample: whole candidates only (the gap features need the full group)
            keepc = rng.random(hi - lo) < sub_rows / len(pm)
            pm = pm[keepc[G["cl"][pm]]]
        c, s = G["c"][pm], G["s"][pm]
        ctx = dict(score=G["score"][pm], nkeys=G["nkeys"][pm], mask=G["mask"][pm],
                   prank=prank[pm], best=G["best"][pm], nopt=np.repeat(G["gl"], G["gl"])[pm])
        return dict(G=G, prank=prank, pm=pm, c=c, s=s, ctx=ctx, X=_stage_b(c, s, ctx, B))

    # ---- 2f-i  fit the ranker (train: cross-fitted by S1 parity) ----
    models = None
    if has_truth:
        t0 = time.time()
        samp = batches[::RANKER_EVERY]
        per = max(RANKER_SAMPLE // max(len(samp), 1), 1)
        XS_, YS_, FO_ = [], [], []
        for lo, hi in samp:
            P = pool_of(lo, hi, sub_rows=per)
            if P is None or len(P["c"]) == 0:
                continue
            y = labels(P["c"] * n1 + P["s"])
            XS_.append(P["X"]); YS_.append(y.astype(np.int8)); FO_.append(P["s"] & 1)
            del P
        X = np.concatenate(XS_); y = np.concatenate(YS_); fo = np.concatenate(FO_)
        del XS_, YS_, FO_
        models = (_fit_ranker(X[fo == 1], y[fo == 1]), _fit_ranker(X[fo == 0], y[fo == 0]))
        ph = np.empty(len(y), np.float32)
        for f in (0, 1):
            if (fo == f).any():
                ph[fo == f] = _predict(models[f], X[fo == f])
        k = min(RANKER_POOLED, len(y))
        pick = rng.choice(len(y), size=k, replace=False)
        R["pooled"] = (X[pick], y[pick])
        log(f"    2f  blocking ranker: {len(y):,} labelled pool pairs (positives {y.mean():.3f}), "
            f"cross-fitted AUC {_auc(y, ph):.4f}   [{time.time()-t0:.1f}s]")
        del X, y, fo, ph

    # ---- 2c..2f  every batch: pairs -> pre-score -> pool -> features -> p -> shortlist ----
    # batches run in S2_THREADS threads; model calls are serialised (LightGBM is multi-threaded)
    t0 = time.time()
    plock = threading.Lock()
    nrh = max(POOL_STRONG, POOL_K_LOW) + 2

    def run(bounds_):
        lo, hi = bounds_
        P = pool_of(lo, hi)
        if P is None:
            return None
        X, c, s = P["X"], P["c"], P["s"]
        with plock:
            if models is None:
                p = _predict(model, X) if model is not None else _fallback_p(X)
            else:
                p = np.empty(len(c), np.float32)
                fo = s & 1
                for f in (0, 1):
                    m = fo == f
                    if m.any():
                        p[m] = _predict(models[f], X[m])
        del X
        prank, _ = _rank_desc(c - lo, p)
        kc = np.where(low[c], KC_MAX_LOW, KC_MAX)
        keep = (prank < kc) & ((p >= P_FLOOR) | (prank == 0))
        G = P["G"]
        o = dict(gen=len(G["uniq"]), pool=len(c), cand=int(keep.sum()))
        lab = None
        if has_truth:
            gid = G["c"] * n1 + G["s"]
            la = labels(gid)
            o["gen_f"] = int(la.sum())
            o["gen_true"] = gid[la]
            o["rank_hist"] = np.bincount(np.minimum(P["prank"][la], nrh - 1), minlength=nrh)
            o["fam_all"] = _fam_counts(G["mask"])
            o["fam_true"] = _fam_counts(G["mask"][la])
            lab = la[P["pm"]]
            o["pool_f"] = int(lab.sum())
            o["cand_f"] = int((keep & lab).sum())
            o["pool_true"] = gid[P["pm"]][lab]
            del gid, la
        idx = np.flatnonzero(keep)
        o["out"] = dict(c=c[idx].astype(np.int32), s=s[idx].astype(np.int32),
                        score=P["ctx"]["score"][idx], nkeys=P["ctx"]["nkeys"][idx],
                        mask=P["ctx"]["mask"][idx], p=p[idx],
                        label=(lab[idx].astype(np.int8) if lab is not None else np.empty(0, np.int8)))
        return o

    OUT = {k: [] for k in ("c", "s", "score", "nkeys", "mask", "p", "label")}
    G_ = dict(gen=0, gen_f=0, pool=0, pool_f=0, cand=0, cand_f=0, sel=0, sel_f=0)
    rank_hist = np.zeros(nrh, np.int64)
    fam_all = np.zeros((2, NF), np.int64)
    fam_true = np.zeros((2, NF), np.int64)
    gen_true, pool_true = [], []
    with cf.ThreadPoolExecutor(max_workers=max(1, S2_THREADS)) as ex:
        for o in ex.map(run, batches):
            if o is None:
                continue
            for k in ("gen", "pool", "cand", "gen_f", "pool_f", "cand_f"):
                G_[k] += o.get(k, 0)
            if has_truth:
                gen_true.append(o["gen_true"]); pool_true.append(o["pool_true"])
                rank_hist += o["rank_hist"]; fam_all += o["fam_all"]; fam_true += o["fam_true"]
            for k, v in o["out"].items():
                OUT[k].append(v)
            del o
    ix = None
    gc.collect()
    sel = {k: (np.concatenate(v) if v else np.empty(0)) for k, v in OUT.items()}
    del OUT
    t_gen = time.time() - t0

    # ---- 2g  context re-score (cross-fitted by S1 parity on train) ----
    t0 = time.time()
    Xk = _ctx_feats(sel["c"], sel["s"], sel["p"], B)
    p1 = sel["p"].astype(np.float32)
    p2 = p1.copy()
    msg_stk = ""
    if has_truth and len(p1):
        y = sel["label"]
        fo = sel["s"] & 1
        stk = [None, None]
        for f in (0, 1):
            m = np.flatnonzero(fo == f)
            if len(m) > STACK_SAMPLE:
                m = np.sort(rng.choice(m, size=STACK_SAMPLE, replace=False))
            if len(m) and y[m].min() != y[m].max():
                stk[f] = _fit_stack(Xk[m], y[m])
        for f in (0, 1):
            m = fo == (1 - f)                        # fitted on fold f -> predicts the other fold
            if stk[f] is not None and m.any():
                p2[m] = _predict(stk[f], Xk[m])
        k = rng.choice(len(y), size=min(STACK_POOLED, len(y)), replace=False)
        R["stk_pooled"] = (Xk[k], y[k])
        msg_stk = f"cross-fitted AUC {_auc(y, p1):.4f} -> {_auc(y, p2):.4f}"
        del stk
    elif stack is not None and len(p1):
        p2 = _predict(stack, Xk)
        msg_stk = "saved model"
    del Xk
    sel["p1"], sel["p"] = p1, p2
    rk, _ = _rank_desc(sel["c"].astype(np.int64), p2)
    sel["prank"] = np.minimum(rk, 32000).astype(np.int16)
    del rk

    # ---- 2h  pair budget: the highest p_block pairs of the country ----
    if TARGET_PER_S1 is not None:
        bps, how = float(TARGET_PER_S1), "fixed"
    elif has_truth:
        bps, how = _auto_budget(p2, sel["label"], n1, R["T"])
    elif budget_saved:
        bps, how = float(budget_saved), "saved by train"
    else:
        bps, how = DEFAULT_PER_S1, "default"
    R["b_used"] = bps
    pick = _budget_pick(p2, int(bps * n1))
    tau = float(p2[pick].min()) if len(pick) else 1.0
    if has_truth:
        R["bcurve"] = _budget_curve(p2, sel["label"], n1, BUDGET_SWEEP)
        R["bcurve1"] = _budget_curve(p1, sel["label"], n1, BUDGET_SWEEP)
        G_["sel_f"] = int(sel["label"][pick].sum())
    G_["sel"] = len(pick)
    sel = {k: (v[pick] if len(v) else v) for k, v in sel.items()}
    log(f"    2c-f pairs {G_['gen']:,} -> pool {G_['pool']:,} ({G_['pool']/n2:.1f}/candidate) -> "
        f"shortlist {G_['cand']:,}   [{t_gen:.1f}s]")
    log(f"    2g-h context re-score ({msg_stk or 'none'}) -> budget {bps:.1f} x S1 ({how}) = "
        f"{G_['sel']:,} pairs (p >= {tau:.4f})   [{time.time()-t0:.1f}s]")

    # ---- S1-side safety cap ----
    fin = np.arange(len(sel["s"]))
    if len(fin):
        r1, _ = _rank_desc(sel["s"].astype(np.int64), sel["p"])
        fin = fin[r1 < TOPK_S1]
    n_fin = len(fin)
    n_fin_f = int(sel["label"][fin].sum()) if has_truth and n_fin else 0
    cov = int(np.unique(sel["s"][fin]).size) if n_fin else 0

    # ---- write this country's rows ----
    t0 = time.time()
    part = ART / f".blk_{split}_{_safe(ctry)}.tsv"
    with open(part, "w") as fh:
        if n_fin:
            o = fin[np.lexsort((-sel["p"][fin], sel["s"][fin]))]
            mk = sel["mask"][o].astype(np.int64)
            df = pd.DataFrame({
                "source1_entity_id": s1_ids[sel["s"][o].astype(np.int64)],
                "candidate_entity_id": cd_ids[sel["c"][o].astype(np.int64)],
                "score": np.round(sel["score"][o], 3),
                "n_keys": sel["nkeys"][o],
                "evidence_mask": mk,
                "n_evidence": _POP[mk & _MASK_ALL],
                "cand_rank": sel["prank"][o],
                "p_block": np.round(sel["p"][o], 4),
            })
            if has_truth:
                df["label"] = sel["label"][o]
            df.to_csv(fh, sep="\t", header=False, index=False)
            del df
    R["part"] = str(part)
    R.update(G_, fin=n_fin, fin_f=n_fin_f, cov=cov, rank_hist=rank_hist, tau=tau,
             fam_all=fam_all, fam_true=fam_true)
    msg = (f"    2h  final {n_fin:,} pairs ({n_fin/max(n1,1):.1f}/S1),  S1 covered {cov:,}/{n1:,}"
           f"   [write {time.time()-t0:.1f}s, country {time.time()-tc0:.1f}s]")
    if has_truth:
        T = max(R["T"], 1)
        msg += (f"\n        recall: generated {G_['gen_f']/T:.4f} -> pool {G_['pool_f']/T:.4f}"
                f" -> shortlist {G_['cand_f']/T:.4f} -> budget {G_['sel_f']/T:.4f} -> final {n_fin_f/T:.4f}"
                f"   precision final {n_fin_f/max(n_fin,1):.4f}")
    log(msg)

    # ---- miss taxonomy (train only) ----
    if has_truth:
        weak = B["weak"]
        gen_ids = np.sort(np.concatenate(gen_true)) if gen_true else np.empty(0, np.int64)
        pool_ids = np.sort(np.concatenate(pool_true)) if pool_true else np.empty(0, np.int64)
        fin_ids = np.sort(sel["c"][fin].astype(np.int64) * n1 + sel["s"][fin])
        stages = (("never generated", tk_all[~_in_sorted(gen_ids, tk_all)]),
                  ("not in pool", gen_ids[~_in_sorted(pool_ids, gen_ids)]),
                  ("not selected", pool_ids[~_in_sorted(fin_ids, pool_ids)]))
        for stage, ids in stages:
            c, s = ids // n1, ids % n1
            o = np.argsort(s, kind="stable")
            rn = np.zeros(len(ids), np.float32); kn = np.zeros(len(ids), np.float32)
            if len(ids):
                _, a = _inter(c[o], s[o], B["R_off"], B["R_ids"], B["R_w"], B["R_SL"], B["R_V"])
                _, b = _inter(c[o], s[o], B["K_off"], B["K_ids"], B["K_w"], B["K_SL"], B["K_V"])
                rn[o], kn[o] = a, b
            wk = weak[c]
            same = (B["raw_c"][c] != 0) & (B["raw_c"][c] == B["raw_s"][s])
            amb = wk & same & (B["name_amb"][s] >= 2)
            unrel = ~wk & (rn == 0) & (kn == 0)
            tr = ~wk & ~unrel & (nonascii[c] if nonascii is not None else False)
            cats = (("ambiguous: no address + name shared by >=2 S1", amb),
                    ("no address (other)", wk & ~amb),
                    ("name unrelated (alias) - address only", unrel),
                    ("transliterated name", tr),
                    ("other", ~wk & ~unrel & ~tr))
            lab_ = np.empty(len(ids), object)
            for name, m in cats:
                R["tax"][(ctry, stage, name)] = int(m.sum())
                lab_[m] = name
            # tie size: S1 records sharing the exact name (same name) or the
            # candidate's first name word (different names) -> how findable it is
            tie = np.where(same, B["name_amb"][s], B["fwdf_c"][c]).astype(np.int64)
            tb = np.searchsorted(np.array([1, 5, 30, 300]), tie, side="left")
            for bi, cnt_ in enumerate(np.bincount(tb, minlength=len(_TIE_LABELS))):
                if cnt_:
                    R["tie"][(ctry, stage, _TIE_LABELS[bi])] = int(cnt_)
            if MISS_CSV and txt is not None and len(ids):
                R["miss_rows"].append(pd.DataFrame({
                    "country": ctry, "stage": stage, "category": lab_,
                    "tie_size": tie, "cand_weak_address": wk,
                    "source1_entity_id": s1_ids[s], "candidate_entity_id": cd_ids[c],
                    "s1_name": _texts(txt[0]["name"], s), "cand_name": _texts(txt[1]["name"], c),
                    "s1_address": _texts(txt[0]["addr"], s), "cand_address": _texts(txt[1]["addr"], c)}))
            if txt is not None and len(ids) and MISS_SAMPLE > 0:
                rs = np.random.default_rng(1)
                for i in rs.choice(len(ids), size=min(max(MISS_SAMPLE // 3, 1), len(ids)), replace=False):
                    R["samples"].append((stage, _row_text(txt[0]["name"], s[i]), _row_text(txt[0]["addr"], s[i]),
                                         _row_text(txt[1]["name"], c[i]), _row_text(txt[1]["addr"], c[i])))
        if R["samples"]:
            _print_samples(ctry, R["samples"], log)
    sel = B = txt = None
    gc.collect()
    return R


def _print_samples(ctry, samples, log):
    cur = None
    for stage, sn, sa, cn, ca in samples:
        if stage != cur:
            log(f"\n    STEP 2 MISSES ({ctry}) — {stage}:")
            cur = stage
        log(f"      S1  : {sn[:44]:<44} | {sa[:52]}")
        log(f"      CAND: {cn[:44]:<44} | {ca[:52]}")


def _country_job(args):
    """process-pool entry point: runs one country, returns its report lines."""
    split, ctry, want_truth = args
    lines = []
    log = lines.append
    truth = None
    if want_truth:
        truth = build_truth(pd.read_csv(ART / FILES[split]["gt"], dtype=str), lambda *_: None)
    bundle = None if want_truth else _load_ranker(log)
    R = _country(split, ctry, truth, bundle, log)
    R["lines"] = lines
    return R


# ============================================================================
#  STEP 2 — all countries + report
# ============================================================================
def step2(split, countries, truth, log, space0, pairs1, recall1):
    t_all = time.time()
    log("\n" + "=" * 100)
    log("STEP 2 — KEYS -> PAIRS -> PRE-SCORE POOL -> PAIR FEATURES -> RANKER -> CONTEXT RE-SCORE -> BUDGET")
    log("=" * 100)
    log(f"  families: {', '.join(FAM)}")
    bdesc = (f"{TARGET_PER_S1} x #S1" if TARGET_PER_S1 is not None else
             f"automatic (recall {TARGET_RECALL} or the shortlist's - {RECALL_SLACK}, "
             f"precision >= {MIN_PRECISION}, <= {MAX_PER_S1} x #S1)")
    log(f"  pool: each candidate's {POOL_K} best pairs by pre-score ({POOL_K_LOW} if low / no address; "
        f"+ strong pairs up to rank {POOL_STRONG});  shortlist: <= {KC_MAX} per candidate "
        f"({KC_MAX_LOW} if low / no address), p >= {P_FLOOR} unless the candidate's best;  "
        f"pair budget {bdesc};  S1 cap {TOPK_S1};  threads {S2_THREADS}")

    has_truth = truth is not None
    bundle = None
    if not has_truth:
        bundle = _load_ranker(log)

    workers = max(1, min(N_WORKERS, len(countries)))
    results = {}
    if workers == 1:
        for ctry in countries:
            results[ctry] = _country(split, ctry, truth, bundle, log)
            gc.collect()
    else:
        log(f"  running {workers} countries in parallel (N_WORKERS={N_WORKERS})")
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_country_job, (split, c, has_truth)): c for c in countries}
            for fu in cf.as_completed(futs):
                R = fu.result()
                for line in R.pop("lines"):
                    log(line)
                results[R["ctry"]] = R

    # ---- merge part files in country order ----
    out_path = ART / f"candidate_pairs_{split}.tsv"
    with open(out_path, "w") as fout:
        fout.write("source1_entity_id\tcandidate_entity_id\tscore\tn_keys\tevidence_mask"
                   "\tn_evidence\tcand_rank\tp_block" + ("\tlabel\n" if has_truth else "\n"))
        for ctry in countries:
            p = results[ctry].get("part")
            if p and os.path.exists(p):
                with open(p) as fin:
                    shutil.copyfileobj(fin, fout, 16 << 20)
                os.remove(p)

    # ---- pooled pre-score + ranker for test (trained on every train country) ----
    if has_truth:
        pooled = [results[c]["pooled"] for c in countries if results[c].get("pooled") is not None]
        ppre = [results[c]["pre_pooled"] for c in countries if results[c].get("pre_pooled") is not None]
        pstk = [results[c]["stk_pooled"] for c in countries if results[c].get("stk_pooled") is not None]
        if pooled:
            import joblib
            t0 = time.time()
            pre = None
            if ppre:
                pre = _fit_pre(np.concatenate([a for a, _, _ in ppre]),
                               np.concatenate([b for _, b, _ in ppre]),
                               np.concatenate([c for _, _, c in ppre]))
            X = np.concatenate([a for a, _ in pooled]); y = np.concatenate([b for _, b in pooled])
            m = _fit_ranker(X, y)
            stk = None
            if pstk:
                Xs = np.concatenate([a for a, _ in pstk]); ys = np.concatenate([b for _, b in pstk])
                if ys.min() != ys.max():
                    stk = _fit_stack(Xs, ys)
                del Xs, ys
            wsum = sum(results[c]["n1"] for c in countries if "b_used" in results[c])
            bud = (sum(results[c]["b_used"] * results[c]["n1"] for c in countries if "b_used" in results[c])
                   / max(wsum, 1)) if wsum else DEFAULT_PER_S1
            bud = round(float(bud), 2)
            RANKER_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(dict(model=m, pre=pre if pre is not None else _pre_default(), stack=stk, budget=bud,
                             features=FEAT_NAMES, fams=FAM, pre_feats=PRE_FEATS, ctx_feats=CTX_FEATS,
                             n_train=len(y)),
                        RANKER_PATH)
            log(f"\n  saved blocking ranker + pre-score + context re-score + pair budget {bud} x S1 for test"
                f" -> {RANKER_PATH}  ({len(y):,} pairs)   [{time.time()-t0:.1f}s]")

    # ------------------------------------------------------------------ report
    Rs = [results[c] for c in countries]
    s1_tot = sum(r["n1"] for r in Rs)
    fin = sum(r.get("fin", 0) for r in Rs)
    ff = sum(r.get("fin_f", 0) for r in Rs)
    cov = sum(r.get("cov", 0) for r in Rs)
    T = max(sum(r["T"] for r in Rs), 1)
    log("\n  " + "-" * 96)
    log("  STEP 2 — OVERALL")
    log("  " + "-" * 96)
    if has_truth:
        g = lambda k: sum(r.get(k, 0) for r in Rs)
        log(f"  {'stage':<46}{'recall':>10}{'precision':>12}{'pairs':>16}{'per S1':>10}")
        for name, p, f in (("2c  pairs sharing >= 1 key (generated)", g("gen"), g("gen_f")),
                           (f"2d  pool (top {POOL_K}/{POOL_K_LOW} by pre-score + strong)", g("pool"), g("pool_f")),
                           (f"2f  ranker shortlist (p >= {P_FLOOR}, <= {KC_MAX}/{KC_MAX_LOW})", g("cand"), g("cand_f")),
                           ("2g-h context re-score + pair budget", g("sel"), g("sel_f")),
                           ("2h  S1 cap -> candidate_pairs", fin, ff)):
            log(f"  {name:<46}{f/T:>10.4f}{f/max(p,1):>12.4f}{p:>16,}{p/max(s1_tot,1):>10.1f}")

        rh = sum(r["rank_hist"] for r in Rs if "rank_hist" in r)
        cum = np.cumsum(rh)
        log("\n  POOL SWEEP — recall if the pool were each candidate's top-K pairs by pre-score")
        log("    " + "".join(f"{'K='+str(k):>9}" for k in POOLK_SWEEP) + f"{'all':>9}")
        log("    " + "".join(f"{cum[min(k, len(cum)) - 1]/T:>9.4f}" for k in POOLK_SWEEP) + f"{g('gen_f')/T:>9.4f}")

        fa = sum(r["fam_all"] for r in Rs if "fam_all" in r)
        ft = sum(r["fam_true"] for r in Rs if "fam_true" in r)
        log("\n  key family (on generated pairs)        recall   precision           pairs   only-this-family")
        for i, f in enumerate(FAM):
            pa, pt = int(fa[0, i]), int(ft[0, i])
            only = int(ft[1, i])
            log(f"    {f:<34}{pt/T:>9.4f}{pt/max(pa,1):>12.4f}{pa:>16,}{only/T:>12.4f}")
        log(f"\n  candidates with >1 true S1 in ground truth: {sum(r.get('multi', 0) for r in Rs):,}")

        log("\n  MISS TAXONOMY (true pairs not in candidate_pairs)")
        log(f"    {'country':<9}{'stage':<17}{'category':<46}{'pairs':>10}{'share of true':>15}")
        amb = 0
        for r in Rs:
            for (c, st, nm), v in r["tax"].items():
                if v == 0:
                    continue
                log(f"    {c:<9}{st:<17}{nm:<46}{v:>10,}{v/T:>15.4f}")
                if nm.startswith("ambiguous"):
                    amb += v
        log(f"    reachable recall (all but the ambiguous misses) = {1 - amb/T:.4f}")

        log("\n  MISSES BY TIE SIZE  (S1 records sharing the exact name, or - when the names differ -")
        log("                      the candidate's first name word;  large = nothing can single out the S1)")
        log(f"    {'country':<9}{'stage':<17}" + "".join(f"{l:>10}" for l in _TIE_LABELS))
        for r in Rs:
            for st_ in ("never generated", "not in pool", "not selected"):
                row = [r["tie"].get((r["ctry"], st_, l), 0) for l in _TIE_LABELS]
                if sum(row):
                    log(f"    {r['ctry']:<9}{st_:<17}" + "".join(f"{v:>10,}" for v in row))
        mr = [d for r in Rs for d in r.get("miss_rows", [])]
        if mr:
            mp = ART / f"blocking_misses_{split}.csv"
            pd.concat(mr, ignore_index=True).to_csv(mp, index=False)
            log(f"    every missed pair -> {mp}")

    log("")
    if has_truth:
        log(f"  STEP 2 RECALL    = {ff/T:.6f}   ({ff:,} of {T:,} true pairs)")
        log(f"  STEP 2 PRECISION = {ff/max(fin,1):.4f}   (1 true pair per {fin/max(ff,1):.2f} candidates)")
    log(f"  candidates       = {fin:,}   ({fin/max(s1_tot,1):.1f} per S1)")
    log(f"  S1 with >=1 cand = {cov:,} / {s1_tot:,}  ({cov/max(s1_tot,1):.2%})")
    log(f"  reduction vs STEP 0 = {1 - fin/space0:.7f}   ({space0/max(fin,1):,.0f}x smaller)")
    if pairs1:
        log(f"  reduction vs STEP 1 = {1 - fin/pairs1:.7f}   ({pairs1/max(fin,1):,.0f}x smaller)")
    if has_truth and recall1:
        log(f"  recall kept vs STEP 1: {ff/T:.6f} of {recall1:.6f}   (difference {ff/T - recall1:+.6f})")

    if has_truth:
        log("\n  SWEEP — pair budget (pairs per S1 = the highest-p pairs of each country; before the S1 cap)")
        log(f"    {'per S1':>7}{'recall':>10}{'precision':>12}{'recall':>10}{'precision':>12}{'pairs':>16}"
            f"{'p_block >= (per country)':>30}")
        log(f"    {'':>7}{'--- ranker only ---':>22}{'--- + re-score ---':>22}")
        for b in BUDGET_SWEEP:
            pk = sum(r["bcurve"][b][0] for r in Rs if "bcurve" in r)
            tk = sum(r["bcurve"][b][1] for r in Rs if "bcurve" in r)
            pk1 = sum(r["bcurve1"][b][0] for r in Rs if "bcurve1" in r)
            tk1 = sum(r["bcurve1"][b][1] for r in Rs if "bcurve1" in r)
            taus = " / ".join(f"{r['bcurve'][b][2]:.4f}" for r in Rs if "bcurve" in r)
            log(f"    {b:>7.1f}{tk1/T:>10.4f}{tk1/max(pk1,1):>12.4f}{tk/T:>10.4f}{tk/max(pk,1):>12.4f}"
                f"{pk:>16,}{taus:>30}")
        used = " / ".join(f"{r['ctry']} {r['b_used']:.1f}" for r in Rs if "b_used" in r)
        log(f"    budget used: {used}   (TARGET_PER_S1 = {TARGET_PER_S1}, TARGET_RECALL = {TARGET_RECALL}, "
            f"MIN_PRECISION = {MIN_PRECISION})")
    log(f"\n  saved -> {out_path}   [STEP 2 {time.time()-t_all:.1f}s]")


# ============================================================================
def main(split="train", run_step1=None):
    t0 = time.time()
    log = print
    if run_step1 is None:
        run_step1 = RUN_STEP1
    log("#" * 100)
    log(f"#  BLOCKING — split = {split}")
    log("#" * 100)

    cfg = FILES[split]
    truth = build_truth(pd.read_csv(ART / cfg["gt"], dtype=str), log) if cfg["gt"] else None
    countries, space0 = step0(split, truth, log)

    pairs1 = rec1 = None
    if run_step1 and truth is not None:
        pairs1, rec1 = step1(split, countries, truth, log, space0)

    step2(split, countries, truth, log, space0, pairs1, rec1)
    log(f"\n  total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    args = sys.argv[1:]
    sp = next((a for a in args if a in FILES), "train")
    for a in args:
        if a.startswith("workers="):
            N_WORKERS = int(a.split("=", 1)[1])
    main(sp, run_step1=True if "step1" in args else (False if "skip1" in args else None))