"""
proposed_model.py  (lecture 6 — proposed model / novelty)
─────────────────────────────────────────────────────────
Fairness-aware models that try to REDUCE the racial FNR gap the baseline inherited.

Two variants, same features & split as the baseline (clean comparison):
  (1) Reweighting MLP   — balanced group×label reweighting: upweights rare race×outcome
                          cells (esp. minority positives) toward equal opportunity (simple, stable).
  (2) Adversarial MLP   — shared encoder + predictor head + RACE adversary connected
                          through a Gradient-Reversal Layer, so the learned
                          representation becomes race-invariant (the novelty).

Task: predict true hypoxemia (SaO2 < 88) from SpO2 + age + sex.
Compares Baseline (LogReg) vs Reweighting vs Adversarial: AUC + FNR by race.

Hardware: tiny tabular model — trains in seconds on M1 CPU. Needs PyTorch:
    pip install torch
Run:  python3 proposed_model.py
"""
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch, torch.nn as nn

torch.manual_seed(42); np.random.seed(42)
DEV = "cpu"
ENR   = "pulseox_pairs_enriched.parquet"
PAIRS = "pulseox_pairs_allicu.parquet"
def log(m): print(m, flush=True)

# ── data (uses enriched vitals if available, else thin baseline features) ──
src = ENR if os.path.exists(ENR) else PAIRS
p = pd.read_parquet(src).dropna(subset=["race", "sex", "spo2", "sao2", "age"]).copy()
p["sex_bin"] = (p["sex"] == "Female").astype(int)
p["y"] = (p["sao2"] < 88).astype(int)
p["race_id"] = p["race"].map({"White": 0, "Black": 1, "Other": 2})
VITALS = [c for c in ["hr", "rr", "mbp", "temp"] if c in p.columns]
FEATURES = ["spo2", "age", "sex_bin"] + VITALS
log(f"source: {'ENRICHED' if src==ENR else 'thin'}  |  features: {FEATURES}")

idx = np.arange(len(p))
tr, tmp = train_test_split(idx, test_size=0.30, stratify=p.y.values, random_state=42)
va, te  = train_test_split(tmp, test_size=0.50, stratify=p.y.values[tmp], random_state=42)

imp = SimpleImputer(strategy="median").fit(p[FEATURES].iloc[tr])
sc  = StandardScaler().fit(imp.transform(p[FEATURES].iloc[tr]))
def prep(ix): return sc.transform(imp.transform(p[FEATURES].iloc[ix]))
Xtr, Xte = prep(tr), prep(te)
ytr, yte = p.y.values[tr], p.y.values[te]
rtr, rte = p.race_id.values[tr], p.race_id.values[te]
log(f"data: train {len(tr):,} | test {len(te):,} | prevalence {100*p.y.mean():.1f}% | features {FEATURES}")

# ── balanced group×label reweighting: each race×outcome cell contributes equally,
#    so rare cells (esp. minority positives) are upweighted -> targets equal FNR ──
def balanced_weights(y, g):
    n = len(y); w = np.ones(n); ncells = len(np.unique(g)) * 2
    for gg in np.unique(g):
        for yy in [0, 1]:
            m = (g == gg) & (y == yy)
            if m.sum() > 0:
                w[m] = n / (ncells * m.sum())     # inverse cell frequency
    w = np.clip(w, 0.2, 8.0)
    return w * len(w) / w.sum()                   # renormalize mean to 1
wtr = balanced_weights(ytr, rtr)

# ── tensors ──
Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=DEV)
ytr_t = torch.tensor(ytr, dtype=torch.float32, device=DEV).unsqueeze(1)
rtr_t = torch.tensor(rtr, dtype=torch.long, device=DEV)
wtr_t = torch.tensor(wtr, dtype=torch.float32, device=DEV).unsqueeze(1)
Xte_t = torch.tensor(Xte, dtype=torch.float32, device=DEV)

# ── Gradient Reversal Layer ──
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd): ctx.lambd = lambd; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return g.neg() * ctx.lambd, None
def grad_reverse(x, lambd): return GradReverse.apply(x, lambd)

# ── network: shared encoder + predictor (+ optional race adversary) ──
class Net(nn.Module):
    def __init__(self, d_in, adversarial=False, lambd=1.0):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
        self.pred = nn.Linear(16, 1)
        self.adversarial = adversarial; self.lambd = lambd
        if adversarial:
            self.adv = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 3))
    def forward(self, x):
        z = self.enc(x)
        yhat = self.pred(z)
        rhat = self.adv(grad_reverse(z, self.lambd)) if self.adversarial else None
        return yhat, rhat, z

def train(adversarial, lambd=1.0, epochs=300, reweight=False):
    net = Net(len(FEATURES), adversarial=adversarial, lambd=lambd).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss(reduction="none"); ce = nn.CrossEntropyLoss()
    tag = "Adversarial" if adversarial else ("Reweighting" if reweight else "Plain")
    for ep in range(epochs):
        net.train(); opt.zero_grad()
        yhat, rhat, z = net(Xtr_t)
        ploss = bce(yhat, ytr_t)
        ploss = (ploss * wtr_t).mean() if reweight else ploss.mean()
        loss = ploss + (ce(rhat, rtr_t) if adversarial else 0.0)
        loss.backward(); opt.step()
        if ep == 0:  # stage 30 — shapes (one forward pass)
            log(f"  [{tag}] shapes — X:{tuple(Xtr_t.shape)} y:{tuple(ytr_t.shape)} "
                f"z:{tuple(z.shape)} pred:{tuple(yhat.shape)}"
                + (f" adv:{tuple(rhat.shape)}" if adversarial else ""))
        if (ep + 1) % 100 == 0:
            log(f"  [{tag}] epoch {ep+1:>3}  pred_loss={ploss.item():.4f}"
                + (f"  adv_loss={ce(rhat, rtr_t).item():.4f}" if adversarial else ""))
    net.eval()
    with torch.no_grad():
        prob = torch.sigmoid(net(Xte_t)[0]).cpu().numpy().ravel()
    return prob

def fnr_by_race(prob, thr=0.5):
    pred = (prob >= thr).astype(int); out = {}
    for g, name in [(0, "White"), (1, "Black"), (2, "Other")]:
        pos = (rte == g) & (yte == 1)
        out[name] = (int(pos.sum()), float(1 - pred[pos].mean()) if pos.sum() > 0 else float("nan"))
    return out
def summarize(name, prob):
    auc = roc_auc_score(yte, prob); fnr = fnr_by_race(prob)
    gap = fnr["Black"][1] - fnr["White"][1]
    log(f"  {name:14s} AUC={auc:.3f} | FNR White={fnr['White'][1]:.2f} Black={fnr['Black'][1]:.2f} "
        f"Other={fnr['Other'][1]:.2f} | Black-White gap={gap:+.2f}")
    return auc, fnr, gap

log("\n── Baseline (LogReg) ──")
lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42).fit(Xtr, ytr)
prob_lr = lr.predict_proba(Xte)[:, 1]

log("\n── Training proposed models (sanity: loss should fall, no NaN/crash) ──")
prob_rw = train(adversarial=False, reweight=True, epochs=300)
# adversarial predictor must ALSO be class-balanced (else it collapses to all-negative
# on the 3.8% positives); gentler lambda keeps the predictor usable.
prob_adv = train(adversarial=True, lambd=0.5, epochs=400, reweight=True)

log("\n" + "=" * 64); log("COMPARISON (test) — AUC and false-negative rate by race"); log("=" * 64)
summarize("Baseline LogReg", prob_lr)
summarize("Reweighting MLP", prob_rw)
summarize("Adversarial MLP", prob_adv)
log("\nGoal: keep AUC reasonable while shrinking the Black-White FNR gap toward 0.")

