#  Amazon ML Challenge 2026 Entity Resolution

Pipeline matching noisy S2/S3 business records to reference S1 entities
(US/India in train, France added in test only). Scored by macro F0.5.

## Pipeline order
1. `0_eda_preprocessing_final.ipynb` — EDA + preprocessing, writes clean CSVs
2. `1_blocking_final.py` — trains/applies the blocking ranker, generates candidate pairs
3. `2_feature_engineering_final.py` — builds pair-level features
4. `3_model_training_final.ipynb` — trains K-fold level-1 models (LightGBM, CatBoost) + graph re-score rounds + decision rule
5. `4_predictions_final.py` — applies the trained pipeline to test data

## Setup
pip install -r requirements.txt

Place competition data under `artifacts/` (not tracked in this repo — see `.gitignore`).
