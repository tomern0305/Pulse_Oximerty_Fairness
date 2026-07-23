"""
enrich_features.py
──────────────────
Adds point-in-time vital signs to each SpO2-SaO2 pair, so the fairness model has
NON-biased signals (heart rate, respiratory rate, mean BP, temperature) to rely on
instead of the racially-biased SpO2 alone.

For each pair (subject, time) we attach the nearest reading of each vital within
±30 min. Output: pulseox_pairs_enriched.parquet (the original pairs + hr/rr/mbp/temp).

Heavy first run (scans chartevents); caches raw vitals so reruns are instant.
Run:  python3 enrich_features.py
"""
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, pandas as pd

BASE = "physionet.org/files/mimiciv/3.1"
ICU = BASE + "/icu"
PAIRS = "pulseox_pairs_allicu.parquet"
RAW_VIT = "raw_vitals.parquet"
OUT = "pulseox_pairs_enriched.parquet"
CHUNK = 500_000
def log(m): print(m, flush=True)

# itemids: HR, RR, mean BP, Temp(C), Temp(F)
ITEMS = {220045: "hr", 220210: "rr", 220181: "mbp", 223762: "tempC", 223761: "tempF"}
RANGE = {"hr": (20, 250), "rr": (0, 60), "mbp": (20, 200), "temp": (30, 43)}

pairs = pd.read_parquet(PAIRS)
pairs["charttime"] = pd.to_datetime(pairs["charttime"])
pair_subjects = set(pairs["subject_id"])

ic = pd.read_csv(ICU + "/icustays.csv.gz", usecols=["subject_id", "stay_id", "intime"])
ic["intime"] = pd.to_datetime(ic["intime"]); intime_stay = ic.set_index("stay_id")["intime"]
stays = set(ic.stay_id)

if os.path.exists(RAW_VIT):
    rv = pd.read_parquet(RAW_VIT); log(f"loaded raw vitals cache: {len(rv):,}")
else:
    log("scanning chartevents for vitals (HR/RR/MBP/Temp), first 24h ...")
    rows = []
    for i, ch in enumerate(pd.read_csv(ICU + "/chartevents.csv.gz", chunksize=CHUNK,
            usecols=["subject_id", "stay_id", "itemid", "valuenum", "charttime"])):
        s = ch[ch.itemid.isin(ITEMS) & ch.stay_id.isin(stays)].dropna(subset=["valuenum"]).copy()
        if len(s):
            s = s[s.subject_id.isin(pair_subjects)]
            if len(s):
                s["charttime"] = pd.to_datetime(s["charttime"])
                s["h"] = (s["charttime"] - s["stay_id"].map(intime_stay)).dt.total_seconds() / 3600
                s = s[(s.h >= 0) & (s.h <= 24)]
                if len(s): rows.append(s[["subject_id", "charttime", "itemid", "valuenum"]])
        if (i + 1) % 50 == 0: log(f"  chartevents ~{(i+1)*CHUNK:,}")
    rv = pd.concat(rows); rv.to_parquet(RAW_VIT, index=False)
    log(f"saved raw vitals: {len(rv):,}")

# build per-vital tables (unify temperature to Celsius)
rv["feat"] = rv["itemid"].map(ITEMS)
def vital_table(feat):
    if feat == "temp":
        c = rv[rv.feat == "tempC"][["subject_id", "charttime", "valuenum"]].copy()
        f = rv[rv.feat == "tempF"][["subject_id", "charttime", "valuenum"]].copy()
        f["valuenum"] = (f["valuenum"] - 32) * 5.0 / 9.0
        t = pd.concat([c, f])
    else:
        t = rv[rv.feat == feat][["subject_id", "charttime", "valuenum"]].copy()
    lo, hi = RANGE[feat]
    t = t[(t.valuenum >= lo) & (t.valuenum <= hi)].rename(columns={"valuenum": feat})
    return t.sort_values("charttime")

enr = pairs.sort_values("charttime").copy()
for feat in ["hr", "rr", "mbp", "temp"]:
    vt = vital_table(feat)
    enr = pd.merge_asof(enr, vt, on="charttime", by="subject_id",
                        tolerance=pd.Timedelta("30min"), direction="nearest")
    log(f"  {feat}: matched {enr[feat].notna().sum():,}/{len(enr):,} pairs ({100*enr[feat].notna().mean():.1f}%)")

enr.to_parquet(OUT, index=False)
log(f"\nsaved enriched pairs -> {OUT}  ({len(enr):,} rows, +4 vital features)")
log("missing rates (%):")
log((enr[["hr", "rr", "mbp", "temp"]].isna().mean() * 100).round(1).to_string())

