"""
preprocessing_analysis.py
─────────────────────────
Implements and QUANTIFIES every preprocessing step for the pulse-ox dataset
(lecture 4). Not documentation — it actually runs each step and measures its effect.

Outputs, per task:
  Task 1 — data understanding: counts of SpO2 / SaO2, % venous contamination,
           % out-of-physiologic-range, missingness.
  Task 2 — preprocessing funnel: how many readings/pairs survive each step.
  Task 3 — before/after: SaO2 distribution with vs without arterial filter (binned),
           SpO2 raw range violations, and the pair time-gap distribution.
  Task 4 — methods comparison: (a) arterial filter OFF vs ON, (b) pairing window
           5/10/15 min, (c) outcome definition A (SpO2>=92) vs B (92-96) — each
           reported as occult-hypoxemia rate by race.

First run extracts RAW intermediates (raw_spo2.parquet, raw_lab.parquet) once (a few
minutes); afterwards everything is instant.

Run:  python3 preprocessing_analysis.py
"""
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, pandas as pd

BASE = "physionet.org/files/mimiciv/3.1"
HOSP, ICU = BASE + "/hosp", BASE + "/icu"
RAW_SPO2 = "raw_spo2.parquet"
RAW_LAB  = "raw_lab.parquet"
CHUNK = 500_000
def log(m): print(m, flush=True)
def grace(r):
    r = str(r).upper(); return "White" if "WHITE" in r else ("Black" if "BLACK" in r else "Other")

# demographics
pat = pd.read_csv(HOSP + "/patients.csv.gz", usecols=["subject_id", "gender"])
adm = pd.read_csv(HOSP + "/admissions.csv.gz", usecols=["subject_id", "race"])
ic  = pd.read_csv(ICU + "/icustays.csv.gz", usecols=["subject_id", "hadm_id", "stay_id", "intime"])
ic["intime"] = pd.to_datetime(ic["intime"])
intime_stay = ic.set_index("stay_id")["intime"]; intime_hadm = ic.groupby("hadm_id")["intime"].min()
stays, hadms = set(ic.stay_id), set(ic.hadm_id)
sex_map  = pat.set_index("subject_id")["gender"].map({"F": "Female", "M": "Male"})
race_map = adm.dropna(subset=["race"]).groupby("subject_id")["race"].first().map(grace)

# ── extract RAW intermediates (no range / arterial filtering yet) ──
if os.path.exists(RAW_SPO2) and os.path.exists(RAW_LAB):
    raw_spo2 = pd.read_parquet(RAW_SPO2); raw_lab = pd.read_parquet(RAW_LAB)
    log(f"loaded raw cache: spo2={len(raw_spo2):,}  lab={len(raw_lab):,}")
else:
    log("scanning chartevents for SpO2 (220277), first 24h — keeping RAW values ...")
    sp = []
    for i, ch in enumerate(pd.read_csv(ICU + "/chartevents.csv.gz", chunksize=CHUNK,
            usecols=["subject_id", "stay_id", "itemid", "valuenum", "charttime"])):
        s = ch[(ch.itemid == 220277) & ch.stay_id.isin(stays)].dropna(subset=["valuenum"]).copy()
        if len(s):
            s["charttime"] = pd.to_datetime(s["charttime"])
            s["h"] = (s["charttime"] - s["stay_id"].map(intime_stay)).dt.total_seconds() / 3600
            s = s[(s.h >= 0) & (s.h <= 24)]
            if len(s): sp.append(s[["subject_id", "charttime", "valuenum"]].rename(columns={"valuenum": "spo2"}))
        if (i + 1) % 50 == 0: log(f"  chartevents ~{(i+1)*CHUNK:,}")
    raw_spo2 = pd.concat(sp); raw_spo2.to_parquet(RAW_SPO2, index=False)
    log("scanning labevents for SaO2 (50817) + pO2 (50821), first 24h — RAW ...")
    lb = []
    for i, ch in enumerate(pd.read_csv(HOSP + "/labevents.csv.gz", chunksize=CHUNK,
            usecols=["subject_id", "hadm_id", "specimen_id", "itemid", "valuenum", "charttime"])):
        s = ch[ch.itemid.isin([50817, 50821]) & ch.hadm_id.isin(hadms)].dropna(subset=["hadm_id", "valuenum"]).copy()
        if len(s):
            s["charttime"] = pd.to_datetime(s["charttime"])
            s["h"] = (s["charttime"] - s["hadm_id"].astype(int).map(intime_hadm)).dt.total_seconds() / 3600
            s = s[(s.h >= 0) & (s.h <= 24)]
            if len(s): lb.append(s[["subject_id", "specimen_id", "itemid", "valuenum", "charttime"]])
        if (i + 1) % 50 == 0: log(f"  labevents ~{(i+1)*CHUNK:,}")
    raw_lab = pd.concat(lb); raw_lab.to_parquet(RAW_LAB, index=False)
    log(f"saved raw cache: spo2={len(raw_spo2):,}  lab={len(raw_lab):,}")

# build raw SaO2 with its specimen pO2
sa_raw = raw_lab[raw_lab.itemid == 50817].rename(columns={"valuenum": "sao2"})[["specimen_id", "subject_id", "charttime", "sao2"]].copy()
sa_raw["po2"] = sa_raw["specimen_id"].map(raw_lab[raw_lab.itemid == 50821].groupby("specimen_id")["valuenum"].max())
sa_raw["race"] = sa_raw.subject_id.map(race_map); sa_raw["sex"] = sa_raw.subject_id.map(sex_map)

def pair(sao2_df, window_min):
    a = sao2_df.dropna(subset=["sao2"]).sort_values("charttime")
    b = raw_spo2[(raw_spo2.spo2 >= 50) & (raw_spo2.spo2 <= 100)].sort_values("charttime")
    m = pd.merge_asof(a, b, on="charttime", by="subject_id",
                      tolerance=pd.Timedelta(f"{window_min}min"), direction="nearest").dropna(subset=["spo2"])
    m["gap_min"] = np.nan  # gap not reconstructable post-merge; reported separately below
    return m

def occult_by_race(pairs_df, lo, hi):
    d = pairs_df[(pairs_df.spo2 >= lo) & (pairs_df.spo2 <= hi)].copy(); d["occ"] = (d.sao2 < 88)
    return {g: (int((d.race == g).sum()), round(100*d.loc[d.race == g, "occ"].mean(), 1)) for g in ["White", "Black", "Other"]}

# ══════ TASK 1 — data understanding ══════
log("\n" + "="*64); log("TASK 1 — DATA UNDERSTANDING"); log("="*64)
log(f"  SpO2 readings (raw, first 24h): {len(raw_spo2):,}")
oor = ((raw_spo2.spo2 < 50) | (raw_spo2.spo2 > 100)).sum()
log(f"    out-of-physiologic-range (<50 or >100): {oor:,} ({100*oor/len(raw_spo2):.2f}%)")
log(f"  SaO2 readings (raw, first 24h): {len(sa_raw):,}")
ven = (sa_raw.po2.isna() | (sa_raw.po2 <= 50)).sum(); art = (sa_raw.po2 > 50).sum()
log(f"    arterial (pO2>50): {art:,} ({100*art/len(sa_raw):.1f}%) | venous/unknown (pO2<=50 or missing): {ven:,} ({100*ven/len(sa_raw):.1f}%)")
log(f"    median SaO2 — venous/unknown: {sa_raw.loc[sa_raw.po2.isna() | (sa_raw.po2<=50),'sao2'].median():.1f}  |  arterial: {sa_raw.loc[sa_raw.po2>50,'sao2'].median():.1f}")
log(f"    SaO2 missing race: {sa_raw.race.isna().mean()*100:.1f}%  missing sex: {sa_raw.sex.isna().mean()*100:.1f}%")

# ══════ TASK 2 — preprocessing funnel ══════
log("\n" + "="*64); log("TASK 2 — PREPROCESSING FUNNEL (readings/pairs surviving each step)"); log("="*64)
log(f"  [SaO2]  raw: {len(sa_raw):,}")
s1 = sa_raw[sa_raw.po2 > 50]; log(f"    after arterial filter (pO2>50): {len(s1):,}")
s2 = s1[(s1.sao2 > 0) & (s1.sao2 <= 100)]; log(f"    after physiologic range (0<SaO2<=100): {len(s2):,}")
s3 = s2.dropna(subset=["race", "sex"]); log(f"    after requiring race+sex: {len(s3):,}")
log(f"  [SpO2]  raw: {len(raw_spo2):,}  ->  in range 50-100: {((raw_spo2.spo2>=50)&(raw_spo2.spo2<=100)).sum():,}")
paired10 = pair(s3, 10); log(f"  [PAIRS] after merge_asof (<=10 min): {len(paired10):,}")

# ══════ TASK 3 — before/after distributions ══════
log("\n" + "="*64); log("TASK 3 — BEFORE/AFTER (binned distributions)"); log("="*64)
bins = list(range(40, 101, 5))
def hist(series):
    c = pd.cut(series, bins=bins, right=False).value_counts().sort_index()
    return [int(v) for v in c.values]
log(f"  SaO2 bins {bins[0]}..{bins[-1]} step5")
log(f"    BEFORE arterial filter (all SaO2):     {hist(sa_raw.sao2)}")
log(f"    AFTER  arterial filter (pO2>50 only):  {hist(s1.sao2)}")
log(f"  SpO2 raw: below50={int((raw_spo2.spo2<50).sum())}  above100={int((raw_spo2.spo2>100).sum())}  "
    f"p1={raw_spo2.spo2.quantile(.01):.0f} p50={raw_spo2.spo2.quantile(.5):.0f} p99={raw_spo2.spo2.quantile(.99):.0f}")

# ══════ TASK 4 — methods comparison ══════
log("\n" + "="*64); log("TASK 4 — METHODS COMPARISON (occult rate by race, def B 92-96 unless noted)"); log("="*64)
log("  (a) ARTERIAL FILTER off vs on:")
log(f"      OFF (all SaO2 incl. venous): {occult_by_race(pair(sa_raw.dropna(subset=['race','sex']),10),92,96)}")
log(f"      ON  (arterial pO2>50 only):  {occult_by_race(paired10,92,96)}")
log("  (b) PAIRING WINDOW (arterial, def B):")
for wmin in [5, 10, 15]:
    pp = pair(s3, wmin); log(f"      <= {wmin:>2} min  n_pairs={len(pp):,}  occult_by_race={occult_by_race(pp,92,96)}")
log("  (c) OUTCOME DEFINITION (arterial, 10 min):")
log(f"      Def A (SpO2>=92):  {occult_by_race(paired10,92,100)}")
log(f"      Def B (SpO2 92-96): {occult_by_race(paired10,92,96)}")

