"""
eval_network_attention_effects.py
=================================
W5 revision: quantify which resting-state networks carry CLASS (disease) signal
rather than SITE (scanner) signal in the post-harmonization GAT attention.

Method (documented, reproducible):
  For each task and each of the 9 networks, using the GAT network-level
  attention weights (a per-subject softmax over the 9 networks):
    class effect = | mean(attention | positive class) - mean(attention | negative class) |
    site  effect = | mean(attention | ADNI)           - mean(attention | TPMIC)            |
  expressed in attention-weight percentage points. A network is "class-dominant"
  when class effect > site effect (its attention tracks diagnosis more than scanner).

Input : results/gat_attention_v2_nolabel.npz (net_attention: 49 x 3 x 9)
Output: results/network_attention_effects.json

Usage:
  conda activate AD
  cd downstream/
  python evaluation/eval_network_attention_effects.py
"""
import re, json
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent.parent
NPZ  = BASE / "results" / "gat_attention_v2_nolabel.npz"
OUT  = BASE / "results" / "network_attention_effects.json"
TASK_CLASSES = {"NC_vs_AD": (0, 2), "NC_vs_MCI": (0, 1), "MCI_vs_AD": (1, 2)}


def detect_site(s):
    return 'ADNI' if re.search(r'\d{3}_S_\d{4}', str(s)) else 'TPMIC'


def main():
    d = np.load(NPZ, allow_pickle=True)
    net = d["net_attention"]               # (49, 3, 9), softmax over 9 networks
    labels = d["labels"]
    sids = d["subject_ids"]
    tasks = list(d["tasks"])
    nets = list(d["network_labels"])
    sites = np.array([detect_site(s) for s in sids])

    results = {}
    for ti, task in enumerate(tasks):
        neg, pos = TASK_CLASSES[task]
        m = np.isin(labels, [neg, pos])
        A = net[m, ti, :] * 100.0          # percentage points
        lab, st = labels[m], sites[m]
        rows = {}
        for ni, nn in enumerate(nets):
            ce = float(abs(A[lab == pos, ni].mean() - A[lab == neg, ni].mean()))
            se = float(abs(A[st == 'ADNI', ni].mean() - A[st == 'TPMIC', ni].mean()))
            rows[nn] = {"class_effect_pp": ce, "site_effect_pp": se,
                        "margin_pp": ce - se, "class_dominant": ce > se}
        class_dom = [n for n in nets if rows[n]["class_dominant"]]
        results[task] = {
            "n": int(m.sum()),
            "n_adni": int(np.sum(st == 'ADNI')),
            "n_tpmic": int(np.sum(st == 'TPMIC')),
            "per_network": rows,
            "class_dominant_networks": class_dom,
        }

    # DMN dominance across tasks
    dmn_all = [t for t in tasks if results[t]["per_network"]["DMN"]["class_dominant"]]
    summary = {"dmn_class_dominant_in_tasks": dmn_all}
    results["_summary"] = summary

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

    # Pretty print
    for task in tasks:
        r = results[task]
        print(f"\n=== {task} (n={r['n']}, ADNI={r['n_adni']}, TPMIC={r['n_tpmic']}) ===")
        print(f"{'net':6s} {'classEff':>9s} {'siteEff':>9s} {'margin':>8s}")
        for nn in nets:
            pn = r["per_network"][nn]
            star = " *" if pn["class_dominant"] else ""
            print(f"{nn:6s} {pn['class_effect_pp']:9.2f} {pn['site_effect_pp']:9.2f} {pn['margin_pp']:8.2f}{star}")
        print(f"  class-dominant: {r['class_dominant_networks']}")
    print(f"\nDMN class-dominant in tasks: {dmn_all}")
    print(f"[SAVED] {OUT}")


if __name__ == "__main__":
    main()
