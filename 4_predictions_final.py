"""4_predictions_final.py — final test inference.

  1. preprocessing   test/test_source{1,2,3}.tsv -> artifacts/test_s{1,2,3}_clean.csv
                     (the SAME prep_final.py functions file 0 uses; skipped when up to date)
  2. candidates      1_blocking_final.py test        -> artifacts/candidate_pairs_test.tsv
  3. pair features   2_feature_engineering_final.py test -> artifacts/pair_features_test.parquet
  4. model           the pipeline in artifacts/models/best_model.json (written by 3_model_training_final.ipynb):
                     level-1 K-fold models (fold mean) -> graph rounds -> decision rule
                     All level-2 features and the rule come from models_final.py — the same code the
                     notebook used on validation.
  5. output          output/matching_results.tsv, output/candidate_pairs.tsv (+ validator if present)

Run from anywhere:  python 4_predictions_final.py            (add --rebuild to redo steps 1-3)
Every step is skipped when its output is newer than its inputs.
"""

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

ART = BASE / "artifacts"
MODEL_DIR = ART / "models"
TEST_DIR = BASE / "test"
OUT_DIR = BASE / "output"

RAW = {i: TEST_DIR / f"test_source{i}.tsv" for i in (1, 2, 3)}
CLEAN = {i: ART / f"test_s{i}_clean.csv" for i in (1, 2, 3)}
PAIRS = ART / "candidate_pairs_test.tsv"
FEATS = ART / "pair_features_test.parquet"
BEST_JSON = MODEL_DIR / "best_model.json"

PREP_FILE = BASE / "prep_final.py"                       # written by 0_eda_preprocessing_final.ipynb
BLOCKING_FILE = BASE / "1_blocking_final.py"
FEATURE_FILE = BASE / "2_feature_engineering_final.py"
MODELS_FILE = BASE / "models_final.py"                   # written by 3_model_training_final.ipynb
VALIDATORS = [BASE / "utils" / "validate_submission.py", BASE / "validate_submission.py"]
ID_COLS = ["source1_entity_id", "candidate_entity_id"]


def banner(text):
    print(f"\n{'=' * 80}\n{text}\n{'=' * 80}", flush=True)


def stale(output, inputs, force=False):
    return force or not output.exists() or any(
        p.exists() and p.stat().st_mtime > output.stat().st_mtime for p in inputs)


def need(path, hint):
    if not path.exists():
        raise FileNotFoundError(f"missing {path}  ->  {hint}")


# -----------------------------------------------------------------------------
# 1. preprocessing — exactly file 0's steps
# -----------------------------------------------------------------------------
def preprocess(force=False):
    banner("1. PREPROCESSING (prep_final.py, as in file 0)")
    todo = [i for i in (1, 2, 3) if stale(CLEAN[i], [RAW[i], PREP_FILE], force)]
    if not todo:
        print("  test_s1/s2/s3_clean.csv: up to date")
        return False
    need(PREP_FILE, "run 0_eda_preprocessing_final.ipynb once (it writes prep_final.py)")
    import prep_final
    for i in todo:
        need(RAW[i], "put the test .tsv files in test/")
        t = time.time()
        df = pd.read_csv(RAW[i], sep="\t", dtype=str, keep_default_na=False, na_values=[""])
        df = prep_final.add_source_column(df)
        df = prep_final.normalize_country(df)
        df = prep_final.process_business_name(df)
        df = prep_final.process_business_address(df)
        df.to_csv(CLEAN[i], index=False)
        print(f"  test_source{i}: {len(df):,} rows, {df.shape[1]} columns -> {CLEAN[i].name}   [{time.time() - t:.0f}s]",
              flush=True)
        del df
        gc.collect()
    return True


# -----------------------------------------------------------------------------
# 2-3. blocking and pair features — run as the scripts themselves
# -----------------------------------------------------------------------------
def run_script(path, *args):
    need(path, "copy it next to this file")
    t = time.time()
    subprocess.run([sys.executable, str(path), *args], cwd=str(BASE), check=True)
    print(f"  {path.name} {' '.join(args)} done   [{time.time() - t:.0f}s]", flush=True)


def generate_candidates(force=False):
    banner("2. CANDIDATE GENERATION (1_blocking_final.py test)")
    if not stale(PAIRS, [*CLEAN.values(), BLOCKING_FILE], force):
        print(f"  {PAIRS.name}: up to date")
        return False
    run_script(BLOCKING_FILE, "test", "skip1")
    need(PAIRS, "blocking did not write the test candidates")
    return True


def build_features(force, required):
    banner("3. PAIR FEATURES (2_feature_engineering_final.py test)")
    import pyarrow.parquet as pq
    rebuild = stale(FEATS, [PAIRS, *CLEAN.values(), FEATURE_FILE], force)
    if not rebuild:
        have = set(pq.read_schema(FEATS).names)
        miss = [f for f in required if f not in have]
        if miss:
            print(f"  {len(miss)} model features missing from {FEATS.name} -> rebuilding")
            rebuild = True
    if not rebuild:
        print(f"  {FEATS.name}: up to date")
        return False
    run_script(FEATURE_FILE, "test")
    need(FEATS, "feature engineering did not write the parquet")
    return True


# -----------------------------------------------------------------------------
# 4. model: level-1 K-fold -> graph rounds -> decision rule
# -----------------------------------------------------------------------------
def load_config():
    need(BEST_JSON, "run 3_model_training_final.ipynb first")
    cfg = json.loads(BEST_JSON.read_text())
    if "pipeline" not in cfg:
        raise ValueError(f"{BEST_JSON} was written by the OLD training notebook — run the new "
                         "3_model_training_final.ipynb (it writes the pipeline + rule this script needs)")
    need(MODELS_FILE, "run 3_model_training_final.ipynb (its first cell writes models_final.py) in this folder")
    return cfg


def predict(cfg):
    import joblib
    import pyarrow.parquet as pq
    import models_final as MF

    banner("4. MODEL")
    feats = cfg["features"]
    print(f"  pipeline : {cfg['name']}  (level-1 {cfg['l1']}, graph rounds {cfg['rounds']})")
    print(f"  validation F0.5 {cfg['val']['f05']:.4f}  precision {cfg['val']['precision']:.4f}"
          f"  recall {cfg['val']['recall']:.4f}")

    t = time.time()
    names = pq.read_schema(FEATS).names
    avail = [f for f in feats if f in names]
    if len(avail) < len(feats):
        print(f"  WARNING: {len(feats) - len(avail)} features not in {FEATS.name} -> NaN: "
              f"{[f for f in feats if f not in names][:10]}")
    cols = ID_COLS + (["country"] if "country" in names else []) + avail
    tb = pq.read_table(FEATS, columns=cols).to_pandas(strings_to_categorical=True)
    s1 = tb["source1_entity_id"].cat.codes.to_numpy().astype(np.int64)
    cand = tb["candidate_entity_id"].cat.codes.to_numpy().astype(np.int64)
    dup = pd.Series(s1 * (int(cand.max()) + 1) + cand).duplicated().to_numpy()
    if dup.any():
        print(f"  {int(dup.sum()):,} duplicate pairs dropped")
        tb = tb.loc[~dup].reset_index(drop=True)
        s1, cand = s1[~dup], cand[~dup]
    S1_CATS = pd.Index(tb["source1_entity_id"].cat.categories)
    CD_CATS = pd.Index(tb["candidate_entity_id"].cat.categories)
    if "country" in tb:
        country = tb["country"].astype(str).to_numpy(dtype=object)
    else:
        country = np.full(len(tb), "all", dtype=object)
    X = np.empty((len(tb), len(feats)), np.float32)
    for j, f in enumerate(feats):
        X[:, j] = tb[f].to_numpy(dtype=np.float32) if f in tb else np.nan
    del tb
    gc.collect()
    print(f"  {len(X):,} pairs x {len(feats)} features, {len(S1_CATS):,} S1s   [{time.time() - t:.0f}s]", flush=True)
    print("  countries:", pd.Series(country).value_counts().to_dict())

    # level-1: mean of each model's K fold models
    P = {}
    for m in cfg["l1"]:
        t = time.time()
        b = joblib.load(MODEL_DIR / f"l1_{m}.joblib")
        if list(b["features"]) != list(feats):
            raise ValueError(f"l1_{m}.joblib was trained on other features than best_model.json lists — rerun file 3")
        P[m] = MF.kfold_proba(b, X)
        print(f"  level-1 {m:<10} ({len(b['models'])} fold models)   [{time.time() - t:.0f}s]", flush=True)
        del b

    if int(cfg["rounds"]) == 0:
        q = np.mean([P[m] for m in cfg["l1"]], axis=0)
    else:
        t = time.time()
        SF, CF = MF.load_fields(ART, "test", S1_CATS, CD_CATS)
        raw = X[:, [feats.index(c) for c in cfg["raw"]]]
        print(f"  graph fields loaded: S1 {sorted(SF)} | candidate {sorted(CF)}   [{time.time() - t:.0f}s]")
        prev = None
        for r in range(1, int(cfg["rounds"]) + 1):
            t = time.time()
            gb = joblib.load(MODEL_DIR / f"graph_r{r}.joblib")
            if list(gb["l1"]) != list(cfg["l1"]) or list(gb["raw_features"]) != list(cfg["raw"]):
                raise ValueError(f"graph_r{r}.joblib does not belong to this best_model.json — rerun file 3")
            p, ex = MF.round_inputs(r, P, prev)
            Z, znames = MF.level2_matrix(p, ex, raw, gb["raw_features"], s1, cand, SF, CF,
                                         gb["anchor_p"], gb["conf_p"])
            if list(znames) != list(gb["feature_names"]):
                raise ValueError(f"graph_r{r}: level-2 feature names differ from training — "
                                 "models_final.py changed? rerun file 3")
            prev = MF.kfold_proba(gb, Z)
            del Z, gb
            gc.collect()
            print(f"  graph round {r}   [{time.time() - t:.0f}s]", flush=True)
        q = prev
    del X
    gc.collect()

    pred = MF.apply_rule(q, s1, cand, cfg["rule"], country)
    return q, pred, s1, cand, S1_CATS, CD_CATS, country


# -----------------------------------------------------------------------------
# 5. output
# -----------------------------------------------------------------------------
def write_output(q, pred, s1, cand, S1_CATS, CD_CATS, country):
    banner("5. OUTPUT")
    cand_ids = np.asarray(CD_CATS, dtype=object)
    s1_ids = np.asarray(S1_CATS, dtype=object)

    def lists(mask):
        idx = np.flatnonzero(mask)
        idx = idx[np.lexsort((-q[idx], s1[idx]))]                  # each S1's ids, most likely first
        g = pd.Series(cand_ids[cand[idx]], index=s1[idx]).groupby(level=0, sort=False).agg(",".join)
        return pd.Series(g.to_numpy(), index=s1_ids[g.index.to_numpy()])

    match = lists(pred)
    cands = lists(np.ones(len(q), bool))
    all_s1 = pd.read_csv(CLEAN[1], usecols=["entity_id"], dtype=str, keep_default_na=False)["entity_id"].drop_duplicates()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"source1_entity_id": all_s1.to_numpy(),
                  "matched_entity_ids": all_s1.map(match).fillna("").to_numpy()}).to_csv(
        OUT_DIR / "matching_results.tsv", sep="\t", index=False)
    pd.DataFrame({"source1_entity_id": all_s1.to_numpy(),
                  "candidate_entity_ids": all_s1.map(cands).fillna("").to_numpy()}).to_csv(
        OUT_DIR / "candidate_pairs.tsv", sep="\t", index=False)

    n_s1 = len(all_s1)
    with_m = int(all_s1.isin(match.index).sum())
    print(f"  S1 entities       : {n_s1:,}   with >= 1 match: {with_m:,} ({with_m / max(n_s1, 1):.1%})")
    print(f"  predicted matches : {int(pred.sum()):,}   of {len(pred):,} candidate pairs")
    for c in pd.unique(country):
        m = country == c
        print(f"    {c:<10} pairs {int(m.sum()):>10,}   matches {int(pred[m].sum()):>9,}")
    print(f"  saved -> {OUT_DIR / 'matching_results.tsv'}")
    print(f"           {OUT_DIR / 'candidate_pairs.tsv'}")


def validate():
    v = next((p for p in VALIDATORS if p.exists()), None)
    if v is None:
        print("\n  validator not found -> skipped")
        return
    banner("6. VALIDATOR")
    subprocess.run([sys.executable, str(v),
                    "--matching", str(OUT_DIR / "matching_results.tsv"),
                    "--candidate", str(OUT_DIR / "candidate_pairs.tsv"),
                    "--test-dir", str(TEST_DIR)], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="redo preprocessing, blocking and features")
    args = ap.parse_args()
    t0 = time.time()
    cfg = load_config()
    changed = preprocess(args.rebuild)
    changed |= generate_candidates(args.rebuild or changed)
    build_features(args.rebuild or changed, cfg["features"])
    write_output(*predict(cfg))
    validate()
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
