"""
eval_discordant_cases.py
========================
Case-study support (#2): find test patients where the two modalities DISAGREE
(fMRI-only and sMRI-only unimodal models predict opposite classes) yet the PCAG
fusion still predicts correctly. Complements the concordant case in the paper
(Fig. case study) by showing the fusion does non-trivial cross-modal arbitration.

For each task it lists discordant-but-correct subjects with their unimodal vs
fusion probabilities and the fMRI GAT network-attention profile.

Usage:
  conda activate AD
  cd downstream/
  python evaluation/eval_discordant_cases.py
"""
import json
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent.parent
RES = BASE / "results"
TASKS = {"NC_vs_AD": (0, 2), "NC_vs_MCI": (0, 1), "MCI_vs_AD": (1, 2)}
LAB = {0: "NC", 1: "MCI", 2: "AD"}


def load(f):
    d = np.load(RES / f, allow_pickle=True)
    return d["test_probs"], d["test_labels"], d["test_subject_ids"]


def main():
    gat = np.load(RES / "gat_attention_v2_nolabel.npz", allow_pickle=True)
    g_sids = list(gat["subject_ids"]); nets = list(gat["network_labels"])
    g_tasks = list(gat["tasks"]); net_att = gat["net_attention"]

    def fmri_networks(sid, task, k=4):
        if sid not in g_sids:
            return []
        si, ti = g_sids.index(sid), g_tasks.index(task)
        na = net_att[si, ti] * 100
        order = np.argsort(na)[::-1][:k]
        return [(nets[i], round(float(na[i]), 1)) for i in order]

    out = {}
    for task, (neg, pos) in TASKS.items():
        fp, fl, fs = load(f"pcag_combat_{task}_probs_v2_fmri_only_fmricombat_nolabel.npz")
        sp, _, ss = load(f"pcag_combat_{task}_probs_v2_smri_only_fmricombat_nolabel.npz")
        ep, _, es = load(f"pcag_combat_{task}_probs_v2_fmricombat_nolabel_ensemble.npz")
        fm = dict(zip(fs, fp)); sm = dict(zip(ss, sp)); em = dict(zip(es, ep))
        cases = []
        for sid, lab in zip(fs, fl):
            if sid not in sm or sid not in em:
                continue
            fpred, spred, fuspred = int(fm[sid] >= .5), int(sm[sid] >= .5), int(em[sid] >= .5)
            if fpred != spred and fuspred == lab:
                cases.append({
                    "subject": str(sid), "true": LAB[pos] if lab else LAB[neg],
                    "fmri_p": round(float(fm[sid]), 3), "fmri_pred": LAB[pos] if fpred else LAB[neg],
                    "smri_p": round(float(sm[sid]), 3), "smri_pred": LAB[pos] if spred else LAB[neg],
                    "fusion_p": round(float(em[sid]), 3), "fusion_pred": LAB[pos] if fuspred else LAB[neg],
                    "fmri_top_networks": fmri_networks(str(sid), task),
                })
        out[task] = cases
        print(f"\n=== {task}: {len(cases)} discordant-but-correct cases ===")
        for c in cases:
            print(f"  {c['subject']:14s} true={c['true']:3s} | fMRI {c['fmri_p']:.2f}->{c['fmri_pred']:3s} "
                  f"sMRI {c['smri_p']:.2f}->{c['smri_pred']:3s} | fusion {c['fusion_p']:.2f}->{c['fusion_pred']:3s}"
                  f" | fMRI nets {c['fmri_top_networks'][:3]}")

    with open(RES / "discordant_cases.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[SAVED] {RES / 'discordant_cases.json'}")


if __name__ == "__main__":
    main()
