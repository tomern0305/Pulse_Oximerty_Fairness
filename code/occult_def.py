"""
occult_def.py
─────────────
Aligns the occult-hypoxemia definition to the literature standard.

The standard (Sjoding 2020, Valbuena 2022) defines occult hypoxemia within the
SpO2 92–96% band (a reassuring-but-borderline reading), NOT SpO2 ≥92 — because
readings of 97–100% are clearly fine and just dilute the rate.

Runs instantly on the cached pulseox_pairs_allicu.parquet (no re-scan).
Prints both definitions side by side, by race and by race × sex, with tests.

Run:  python3 occult_def.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from math import sqrt, erf

PAIRS = "pulseox_pairs_allicu.parquet"
p = pd.read_parquet(PAIRS)
p["grp"] = p["race"] + " " + p["sex"]

def zt(x1, n1, x2, n2):
    if n1 == 0 or n2 == 0: return float("nan"), float("nan"), float("nan")
    a, b = x1/n1, x2/n2; pp = (x1+x2)/(n1+n2); se = sqrt(pp*(1-pp)*(1/n1+1/n2))
    z = (b-a)/se if se > 0 else 0
    return a, b, 2*(1-0.5*(1+erf(abs(z)/sqrt(2))))

def report(d, title):
    d = d.copy(); d["occ"] = (d.sao2 < 88).astype(int)
    print("\n" + "=" * 64); print(title + f"   (n={len(d):,} readings in band)"); print("=" * 64)
    print("  by race:")
    for g in ["White", "Black", "Other"]:
        s = d[d.race == g]
        if len(s): print(f"    {g:6s} n={len(s):>6,}  occult={100*s.occ.mean():.1f}%  (events={int(s.occ.sum())})")
    sW, sB = d[d.race == "White"], d[d.race == "Black"]
    a, b, pv = zt(int(sW.occ.sum()), len(sW), int(sB.occ.sum()), len(sB))
    rr = (b/a) if a > 0 else float("nan")
    print(f"    Black vs White: {b*100:.1f}% vs {a*100:.1f}%  RR={rr:.2f}  p={pv:.4f}")
    print("  by race x sex:")
    for g in ["White Female", "Black Female", "White Male", "Black Male", "Other Female", "Other Male"]:
        s = d[d.grp == g]
        if len(s): print(f"    {g:14s} n={len(s):>6,}  occult={100*s.occ.mean():.1f}%  (events={int(s.occ.sum())})")
    for w, bk in [("White Female", "Black Female"), ("White Male", "Black Male")]:
        sa, sb = d[d.grp == w], d[d.grp == bk]
        if len(sa) > 5 and len(sb) > 5:
            a, b, pv = zt(int(sa.occ.sum()), len(sa), int(sb.occ.sum()), len(sb))
            print(f"    {bk} ({b*100:.1f}%) vs {w} ({a*100:.1f}%)  p={pv:.4f}  {'SIGNIFICANT' if pv<0.05 else 'ns'}")

report(p[p.spo2 >= 92],                       "DEFINITION A — SpO2 >= 92 (what we used so far)")
report(p[(p.spo2 >= 92) & (p.spo2 <= 96)],    "DEFINITION B — SpO2 92-96 (literature standard)")

