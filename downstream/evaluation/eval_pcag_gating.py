"""
eval_pcag_gating.py
===================
Fusion-mechanism analysis (#3): how much does PCAG draw on each modality?

PCAG fuses fMRI (Query) and sMRI (Key/Value) with a per-patient sigmoid gate
    S = sigmoid( (W_Q e_f) * (W_K e_s) * P ),   V_hat = S * (W_V e_s)
where S in (0,1)^{d_f} modulates how much of the sMRI Value passes into the
fused representation (fMRI always contributes via the residual ReLU(Q)).
mean(S) is therefore a per-patient index of sMRI utilisation: higher = the
fusion opens the structural gate wider.

We extract S for every test subject (averaged over the task's ensemble) and
report mean sMRI-gate openness per task and per class. Expectation: MCI vs. AD
opens the gate widest, consistent with sMRI being essential there (fMRI-only
collapses to chance on that task).

Usage:
  conda activate AD
  cd downstream/
  python evaluation/eval_pcag_gating.py
"""
import sys, re, json
import numpy as np
import pandas as pd
import torch
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
import inference_pipeline_v2 as P

LAB = {0: "NC", 1: "MCI", 2: "AD"}
TASKS = {"NC_vs_AD": (0, 2), "NC_vs_MCI": (0, 1), "MCI_vs_AD": (1, 2)}


def detect_site(s):
    return 'ADNI' if re.search(r'\d{3}_S_\d{4}', str(s)) else 'TPMIC'


@torch.no_grad()
def gate_for(pcag, fmri_emb, smri_t):
    """Recompute the PCAG sigmoid gate S = sigmoid((Q*K)*P) and return mean(S)."""
    Q = pcag.fmri_proj(fmri_emb)
    K = pcag.smri_proj_k(smri_t)
    Pp = (torch.tanh(Q) * torch.tanh(K) + 1) / 2
    S = torch.sigmoid((Q * K) * Pp)
    return float(S.mean().item())


def main():
    fmri_combat = P.load_fmri_combat_estimates()
    smri_all = pd.read_csv(BASE / "brainiac_features_combined_test_v2.csv")
    cols = [f"Feature_{i}" for i in range(768)]
    df_all = pd.read_csv(BASE / "pcag_test_aligned_v2.csv")

    groups = {
        'NC_vs_AD':  P.load_pcag_seed_groups('NC_vs_AD',  P.PCAG_NCAD_DIRS),
        'NC_vs_MCI': P.load_pcag_seed_groups('NC_vs_MCI', P.PCAG_NCMCI_DIRS),
        'MCI_vs_AD': P.load_pcag_seed_groups('MCI_vs_AD', P.PCAG_MCIAD_ENSEMBLE_DIRS),
    }
    out = {}
    for task, (neg, pos) in TASKS.items():
        sc = P.load_combat_params(task)
        models = [m for grp in groups[task] for m in grp]   # flatten ensemble
        df = df_all[df_all["label"].isin([neg, pos])].reset_index(drop=True)
        per_subj = []
        for _, r in df.iterrows():
            site = detect_site(r["subject_id"])
            mat = np.load(r["matrix_path"]); mat = mat[0] if mat.ndim == 3 else mat
            mat = ((mat + mat.T) / 2).astype(np.float32)
            mat = P.apply_combat_fmri(mat, site, fmri_combat)
            x = torch.tensor(P.extract_node_features(mat)).unsqueeze(0).to(P.DEVICE)
            adj = torch.tensor(P.build_adj(mat)).unsqueeze(0).to(P.DEVICE)
            sf = P.apply_combat(smri_all.iloc[int(r["smri_feat_row"])][cols].values.astype(np.float32), site, sc)
            st = torch.tensor(sf, dtype=torch.float32).unsqueeze(0).to(P.DEVICE)
            gates = []
            for m in models:
                with torch.no_grad():
                    femb = m.fmri_encoder(x, adj)
                gates.append(gate_for(m.pcag, femb, st))
            per_subj.append({"label": int(r["label"]), "gate": float(np.mean(gates))})
        gates = np.array([s["gate"] for s in per_subj])
        labels = np.array([s["label"] for s in per_subj])
        out[task] = {
            "mean_gate": float(gates.mean()),
            "std_gate": float(gates.std()),
            "gate_pos_class": float(gates[labels == pos].mean()),
            "gate_neg_class": float(gates[labels == neg].mean()),
            "n": int(len(gates)),
        }
        print(f"{task:11s} mean sMRI-gate S = {gates.mean():.3f} ± {gates.std():.3f}  "
              f"| {LAB[neg]}={out[task]['gate_neg_class']:.3f}  {LAB[pos]}={out[task]['gate_pos_class']:.3f}")

    ranked = sorted(out, key=lambda t: out[t]["mean_gate"], reverse=True)
    print(f"\nsMRI-gate openness ranking (widest first): "
          + " > ".join(f"{t}({out[t]['mean_gate']:.3f})" for t in ranked))
    with open(BASE / "results" / "pcag_gating.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[SAVED] {BASE / 'results' / 'pcag_gating.json'}")


if __name__ == "__main__":
    main()
