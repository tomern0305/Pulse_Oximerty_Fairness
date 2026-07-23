"""
pulseox_allicu.py
─────────────────
Widened pulse-oximetry bias study over ALL ICU patients (not just sepsis),
first 24h of each stay. Greatly increases sample size — especially the small
intersectional cells (e.g., Black women) the lecturer cares about.

Outputs occult-hypoxemia + device over-read by race and by race × sex, with tests.
First run extracts + caches pulseox_pairs_allicu.parquet (a few minutes);
later runs load it instantly.

Run:  python3 pulseox_allicu.py

Note: this keeps SpO2 for ALL ICU stays (~a few million rows in the first 24h) —
heavier than the sepsis version but fine on a normal laptop.
"""
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, pandas as pd
from math import sqrt, erf

BASE = "physionet.org/files/mimiciv/3.1"
HOSP, ICU = BASE + "/hosp", BASE + "/icu"
PAIRS = "pulseox_pairs_allicu.parquet"
CHUNK = 500_000
def log(m): print(m, flush=True)

def grace(r):
    r = str(r).upper()
    return "White" if "WHITE" in r else ("Black" if "BLACK" in r else "Other")

if os.path.exists(PAIRS):
    pairs = pd.read_parquet(PAIRS); log(f"loaded cached pairs: {len(pairs):,}")
else:
    log("loading demographics (patients, admissions, icustays — ALL ICU) ...")
    pat = pd.read_csv(HOSP + "/patients.csv.gz", usecols=["subject_id", "gender", "anchor_age"])
    adm = pd.read_csv(HOSP + "/admissions.csv.gz", usecols=["subject_id", "hadm_id", "race"])
    ic  = pd.read_csv(ICU + "/icustays.csv.gz", usecols=["subject_id", "hadm_id", "stay_id", "intime"])
    ic["intime"] = pd.to_datetime(ic["intime"])
    intime_stay = ic.set_index("stay_id")["intime"]
    intime_hadm = ic.groupby("hadm_id")["intime"].min()
    stays, hadms = set(ic.stay_id), set(ic.hadm_id)
    sex_map  = pat.set_index("subject_id")["gender"].map({"F": "Female", "M": "Male"})
    age_map  = pat.set_index("subject_id")["anchor_age"]
    race_map = adm.dropna(subset=["race"]).groupby("subject_id")["race"].first().map(grace)
    log(f"  {len(stays):,} ICU stays")

    log("scanning chartevents for SpO2 (220277), first 24h, ALL stays ...")
    sp = []
    for i, ch in enumerate(pd.read_csv(ICU + "/chartevents.csv.gz", chunksize=CHUNK,
            usecols=["subject_id", "stay_id", "itemid", "valuenum", "charttime"])):
        s = ch[(ch.itemid == 220277) & ch.stay_id.isin(stays)].copy()
        if len(s):
            s["charttime"] = pd.to_datetime(s["charttime"])
            s["h"] = (s["charttime"] - s["stay_id"].map(intime_stay)).dt.total_seconds() / 3600
            s = s[(s.h >= 0) & (s.h <= 24) & (s.valuenum >= 50) & (s.valuenum <= 100)]
            if len(s): sp.append(s[["subject_id", "charttime", "valuenum"]].rename(columns={"valuenum": "spo2"}))
        if (i + 1) % 50 == 0: log(f"  chartevents ~{(i+1)*CHUNK:,}")
    spo2 = pd.concat(sp); log(f"  SpO2 readings: {len(spo2):,}")

    log("scanning labevents for SaO2 (50817) + pO2 (50821), first 24h ...")
    lb = []
    for i, ch in enumerate(pd.read_csv(HOSP + "/labevents.csv.gz", chunksize=CHUNK,
            usecols=["subject_id", "hadm_id", "specimen_id", "itemid", "valuenum", "charttime"])):
        s = ch[ch.itemid.isin([50817, 50821]) & ch.hadm_id.isin(hadms)].copy()
        if len(s):
            s = s.dropna(subset=["hadm_id", "valuenum"]); s["charttime"] = pd.to_datetime(s["charttime"])
            s["h"] = (s["charttime"] - s["hadm_id"].astype(int).map(intime_hadm)).dt.total_seconds() / 3600
            s = s[(s.h >= 0) & (s.h <= 24)]
            if len(s): lb.append(s[["subject_id", "specimen_id", "itemid", "valuenum", "charttime"]])
        if (i + 1) % 50 == 0: log(f"  labevents ~{(i+1)*CHUNK:,}")
    lab = pd.concat(lb)
    sa = lab[lab.itemid == 50817].rename(columns={"valuenum": "sao2"})[["specimen_id", "subject_id", "charttime", "sao2"]]
    sa["po2"] = sa["specimen_id"].map(lab[lab.itemid == 50821].groupby("specimen_id")["valuenum"].max())
    sao2 = sa[(sa.po2 > 50) & (sa.sao2 > 0) & (sa.sao2 <= 100)].sort_values("charttime")
    spo2 = spo2.sort_values("charttime")
    log(f"  arterial SaO2 readings: {len(sao2):,}")

    pairs = pd.merge_asof(sao2, spo2, on="charttime", by="subject_id",
                          tolerance=pd.Timedelta("10min"), direction="nearest").dropna(subset=["spo2"])
    pairs["race"] = pairs["subject_id"].map(race_map)
    pairs["sex"]  = pairs["subject_id"].map(sex_map)
    pairs["age"]  = pairs["subject_id"].map(age_map)
    pairs = pairs.dropna(subset=["race", "sex"])
    pairs.to_parquet(PAIRS, index=False); log(f"  saved {len(pairs):,} pairs -> {PAIRS}")

log(f"\nPaired readings (all-ICU): {len(pairs):,}")
pairs["grp"] = pairs["race"] + " " + pairs["sex"]
dev = pairs[pairs.spo2 >= 92].copy(); dev["occult"] = (dev.sao2 < 88).astype(int)

def zt_prop(x1, n1, x2, n2):
    p1, p2 = x1/n1, x2/n2; pp = (x1+x2)/(n1+n2); se = sqrt(pp*(1-pp)*(1/n1+1/n2))
    z = (p2-p1)/se if se > 0 else 0; return p1, p2, 2*(1-0.5*(1+erf(abs(z)/sqrt(2))))
def zt_mean(a, b):
    se = sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b)); z = (b.mean()-a.mean())/se if se > 0 else 0
    return a.mean(), b.mean(), 2*(1-0.5*(1+erf(abs(z)/sqrt(2))))

log("\n=== Occult hypoxemia by race ===")
for g in ["White", "Black", "Other"]:
    s = dev[dev.race == g]
    if len(s): log(f"  {g:6s} n={len(s):>6,}  occult={100*s.occult.mean():.1f}%")
p1, p2, pv = zt_prop(int(dev[dev.race=='White'].occult.sum()), (dev.race=='White').sum(),
                     int(dev[dev.race=='Black'].occult.sum()), (dev.race=='Black').sum())
log(f"  Black vs White: {p2*100:.1f}% vs {p1*100:.1f}%  p={pv:.4f}")

log("\n=== Occult hypoxemia by race x sex (intersectional) ===")
groups = ["White Female", "Black Female", "White Male", "Black Male", "Other Female", "Other Male"]
for g in groups:
    s = dev[dev.grp == g]
    if len(s): log(f"  {g:14s} n={len(s):>6,}  occult={100*s.occult.mean():.1f}%  (events={int(s.occult.sum())})")
for w, b in [("White Female", "Black Female"), ("White Male", "Black Male")]:
    sa_, sb_ = dev[dev.grp == w], dev[dev.grp == b]
    if len(sa_) > 5 and len(sb_) > 5:
        p1, p2, pv = zt_prop(int(sa_.occult.sum()), len(sa_), int(sb_.occult.sum()), len(sb_))
        log(f"  {b} ({p2*100:.1f}%) vs {w} ({p1*100:.1f}%)  p={pv:.4f}  {'SIGNIFICANT' if pv<0.05 else 'ns'}")

log("\n=== Device over-read (SpO2 - SaO2) by race x sex  [continuous = higher power] ===")
pairs["bias"] = pairs.spo2 - pairs.sao2
for g in groups:
    s = pairs[pairs.grp == g]
    if len(s): log(f"  {g:14s} n={len(s):>6,}  mean={s.bias.mean():+.2f}  sd={s.bias.std():.2f}")
for w, b in [("White Female", "Black Female"), ("White Male", "Black Male")]:
    a, c = pairs[pairs.grp == w].bias, pairs[pairs.grp == b].bias
    if len(a) > 5 and len(c) > 5:
        m1, m2, pv = zt_mean(a, c)
        log(f"  {b} (+{m2:.2f}) vs {w} (+{m1:.2f})  p={pv:.4f}  {'SIGNIFICANT' if pv<0.05 else 'ns'}")

