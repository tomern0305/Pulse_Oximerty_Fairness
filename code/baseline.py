"""
baseline.py  (lecture 5 — Baseline / Benchmark)
───────────────────────────────────────────────
Baseline model for the pulse-ox project: a simple Logistic Regression that predicts
TRUE hypoxemia (SaO2 < 88%) from the pulse-oximeter reading + basic demographics.

Purpose: a literature-standard, interpretable reference point, and a test of whether
a model that relies on SpO2 INHERITS the device's racial bias (higher false-negative
rate for Black patients). Benchmarked against the naive clinical rule (flag if SpO2<92).

Runs instantly on the cached pulseox_pairs_allicu.parquet (built earlier). No re-scan.

Run:  python3 baseline.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

PAIRS = "pulseox_pairs_allicu.parquet"
def log(m): print(m, flush=True)

p = pd.read_parquet(PAIRS).dropna(subset=["race", "sex", "spo2", "sao2", "age"]).copy()
p["sex_bin"] = (p["sex"] == "Female").astype(int)
p["y"] = (p["sao2"] < 88).astype(int)              # true hypoxemia (arterial)

FEATURES = ["spo2", "age", "sex_bin"]               # baseline features (no race)
log("="*64); log("BASELINE — predict true hypoxemia (SaO2<88) from SpO2 + age + sex"); log("="*64)
log(f"  pairs: {len(p):,}  |  true-hypoxemia prevalence (class imbalance): {100*p.y.mean():.1f}%  (positives={int(p.y.sum()):,})")

# ── train / val / test split (70/15/15, stratified) ──
idx = np.arange(len(p))
tr, tmp = train_test_split(idx, test_size=0.30, stratify=p.y.values, random_state=42)
va, te  = train_test_split(tmp, test_size=0.50, stratify=p.y.values[tmp], random_state=42)
log(f"  split — train {len(tr):,} | val {len(va):,} | test {len(te):,}")

Xtr, ytr = p[FEATURES].iloc[tr], p.y.values[tr]
Xte, yte = p[FEATURES].iloc[te], p.y.values[te]
race_te  = p["race"].values[te]

# ── baseline model: Logistic Regression (class_weight balanced for imbalance) ──
model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
model.fit(Xtr, ytr)
prob = model.predict_proba(Xte)[:, 1]

auc = roc_auc_score(yte, prob); pr = average_precision_score(yte, prob)
log("\n── Evaluation (test set) ──")
log(f"  Overall AUC = {auc:.3f}   PR-AUC = {pr:.3f}   (baseline rate {100*yte.mean():.1f}%)")

# ── fairness: recall / FNR by race at threshold 0.5 ──
pred = (prob >= 0.5).astype(int)
def fnr_recall(mask):
    pos = mask & (yte == 1)
    if pos.sum() == 0: return np.nan, np.nan, 0
    rec = (pred[pos] == 1).mean()
    return rec, 1 - rec, int(pos.sum())
log("\n── Fairness — does the SpO2-based model miss true-hypoxemic patients more for some groups? ──")
log("  (FNR = fraction of truly hypoxemic patients the model MISSED)")
for g in ["White", "Black", "Other"]:
    rec, fnr, n = fnr_recall(race_te == g)
    log(f"    {g:6s} true-hypoxemic n={n:>4}  recall={rec:.2f}  FNR={fnr:.2f}")

# ── benchmark: naive clinical rule (flag hypoxemia if SpO2 < 92) ──
naive = (Xte["spo2"].values < 92).astype(int)
log("\n── Benchmark: naive clinical rule (SpO2 < 92  ->  hypoxemic) ──")
log(f"  Overall: flags {100*naive.mean():.1f}% of readings")
for g in ["White", "Black", "Other"]:
    pos = (race_te == g) & (yte == 1)
    if pos.sum() > 0:
        fnr = (naive[pos] == 0).mean()
        log(f"    {g:6s} true-hypoxemic n={int(pos.sum()):>4}  FNR(missed)={fnr:.2f}")

