# -*- coding: utf-8 -*-
"""Error analysis of the baseline model (Discussion, Part 7).

Takes the truly hypoxemic patients in the test set, splits them into the cases
the baseline caught and the cases it missed (false negatives), and characterizes
the misses. Produces the numbers quoted in the paper and the figure fig_errors.png.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

PAIRS = "pulseox_pairs_allicu.parquet"
FEATURES = ["spo2", "age", "sex_bin"]      # thin baseline that inherits the device bias
OUT_FIG = "fig_errors.png"

# ---- load and label ----------------------------------------------------------
p = pd.read_parquet(PAIRS).dropna(subset=["race", "sex", "spo2", "sao2", "age"]).copy()
p["sex_bin"] = (p.sex == "Female").astype(int)
p["y"] = (p.sao2 < 88).astype(int)         # true hypoxemia is the target

# ---- same split as the modeling section (seed 42) ----------------------------
idx = np.arange(len(p))
tr, tmp = train_test_split(idx, test_size=.3, stratify=p.y, random_state=42)
va, te = train_test_split(tmp, test_size=.5, stratify=p.y.values[tmp], random_state=42)

model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
).fit(p[FEATURES].iloc[tr], p.y.values[tr])

tt = p.iloc[te].copy()
tt["prob"] = model.predict_proba(p[FEATURES].iloc[te])[:, 1]
tt["pred"] = (tt.prob >= .5).astype(int)

pos = tt[tt.y == 1]                          # truly hypoxemic in the test set
fn = pos[pos.pred == 0]                      # missed (false negatives)
tp = pos[pos.pred == 1]                      # caught

# ---- pattern statistics quoted in the paper ----------------------------------
print("truly hypoxemic in test: %d | missed (FN): %d (%.0f%%) | caught (TP): %d"
      % (len(pos), len(fn), 100 * len(fn) / len(pos), len(tp)))
print("\nPattern 1  device over-read (SpO2 - SaO2)")
print("  missed median = %.1f | caught median = %.1f"
      % ((fn.spo2 - fn.sao2).median(), (tp.spo2 - tp.sao2).median()))
print("  missed with SpO2 > 96 (reassuring): %.0f%%" % (100 * (fn.spo2 > 96).mean()))
print("\nPattern 2  severity of the missed cases (SaO2)")
print("  missed with SaO2 < 85 (severe): %.0f%% | SaO2 85-88 (borderline): %.0f%%"
      % (100 * (fn.sao2 < 85).mean(), 100 * fn.sao2.between(85, 88).mean()))
print("\nPattern 3  racial gradient in the miss rate (FNR)")
for g in ["White", "Black", "Other"]:
    sub = pos[pos.race == g]
    print("  %-6s FNR = %.2f  (n=%d hypoxemic)" % (g, 1 - sub.pred.mean(), len(sub)))

# ---- figure ------------------------------------------------------------------
NAVY, TEAL, CORAL, GREY = "#0B5560", "#0E7C86", "#E07A5F", "#8A8A8A"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.9))

a = ax[0]
cr, mr = (tp.spo2 - tp.sao2).values, (fn.spo2 - fn.sao2).values
parts = a.boxplot([cr, mr], vert=True, patch_artist=True, widths=0.55,
                  showfliers=False, medianprops=dict(color="white", linewidth=2))
for patch, c in zip(parts["boxes"], [TEAL, CORAL]):
    patch.set_facecolor(c); patch.set_alpha(0.9)
for i, (d, c) in enumerate(zip([cr, mr], [TEAL, CORAL]), start=1):
    a.scatter(np.random.normal(i, 0.06, len(d)), d, s=9, color=c, alpha=0.35,
              edgecolors="none", zorder=3)
a.set_xticks([1, 2]); a.set_xticklabels(["Caught\n(detected)", "Missed\n(false negative)"])
a.set_ylabel("Device over-read  SpO2 - SaO2  (points)"); a.set_xlim(0.5, 2.7)
a.set_title("A. Missed cases have a far larger over-read", fontsize=10.5, color=NAVY, loc="left", pad=8)
a.axhline(0, color=GREY, lw=0.8, ls="--"); a.spines[["top", "right"]].set_visible(False)
a.text(1.32, np.median(cr), "median %.1f" % np.median(cr), ha="left", va="center", color=TEAL, fontsize=8.5)
a.text(2.32, np.median(mr), "median %.1f" % np.median(mr), ha="left", va="center", color=CORAL, fontsize=8.5)

b = ax[1]
races = ["White", "Black", "Other"]
vals = [1 - pos[pos.race == g].pred.mean() for g in races]
bars = b.bar(races, vals, color=[TEAL, CORAL, GREY], alpha=0.9, width=0.62)
for bar, v in zip(bars, vals):
    b.text(bar.get_x() + bar.get_width() / 2, v + 0.015, "%.0f%%" % (100 * v),
           ha="center", color=NAVY, fontsize=9.5)
b.set_ylim(0, 0.62); b.set_ylabel("Missed rate among truly hypoxemic (FNR)")
b.set_title("B. The model misses more Black and Other patients", fontsize=10.5, color=NAVY, loc="left", pad=8)
b.spines[["top", "right"]].set_visible(False)

plt.tight_layout(); plt.savefig(OUT_FIG, dpi=160, bbox_inches="tight")
print("\nsaved", OUT_FIG)
