"""results/eval.json -> results/summary.md, results/target_sweep.png, results/loss.png"""
import json, glob, os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dtlib import load_trajs, parse_name, normalized_score

res = json.load(open("results/eval.json"))
DS = ["hopper-medium", "halfcheetah-medium", "walker2d-medium", "hopper-simple", "halfcheetah-simple", "walker2d-simple", "hopper-medium-expert"]
DS = [d for d in DS if any(v["dataset"] == d for v in res.values())]
ms = lambda x: f"{np.mean(x):.1f} +- {np.std(x, ddof=1):.1f}" if len(x) > 1 else f"{x[0]:.1f}"

def per_seed(ds, algo, frac=None):
    out = []
    for v in res.values():
        if v["dataset"] == ds and v["algo"] == algo:
            norm = v["norm"] if algo != "dt" else next(t for t in v["by_target"].values() if abs(t["frac"] - frac) < 1e-9)["norm"]
            out.append(float(np.mean(norm)))
    return out

lines = ["| Dataset | data mean | data max | BC | %BC (top 10%) | DT, target = 1.0 x data max | DT, best target (oracle) | seeds |", "|---|---|---|---|---|---|---|---|"]
stats = {}
for ds in DS:
    env, _ = parse_name(ds)
    R = np.array([t["rew"].sum() for t in load_trajs(ds)])
    stats[ds] = (normalized_score(env, R.mean()), normalized_score(env, R.max()))
    fr = sorted({t["frac"] for v in res.values() if v["dataset"] == ds and v["algo"] == "dt" for t in v["by_target"].values()})
    best = max(fr, key=lambda f: np.mean(per_seed(ds, "dt", f)))
    bc, pbc, dt, dtb = per_seed(ds, "bc"), per_seed(ds, "pbc"), per_seed(ds, "dt", 1.0), per_seed(ds, "dt", best)
    lines.append(f"| {ds} | {stats[ds][0]:.1f} | {stats[ds][1]:.1f} | {ms(bc)} | {ms(pbc)} | {ms(dt)} | {ms(dtb)} (x{best}) | {len(dt)} |")
    lines.append("")  # placeholder removed below
lines = [l for l in lines if l]
open("results/summary.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# per-seed raw lines
detail = ["", "Per-seed means (normalized):"]
for ds in DS:
    for algo, fr in (("bc", None), ("pbc", None), ("dt", 1.0)):
        detail.append(f"{ds} {algo}: " + ", ".join(f"{x:.1f}" for x in per_seed(ds, algo, fr)))
open("results/summary.md", "a").write("\n".join(detail) + "\n")

n = len(DS); cols = 4; rows = (n + cols - 1) // cols
fig, axs = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
for ax, ds in zip(axs.flat, DS):
    env, _ = parse_name(ds)
    runs = [v for v in res.values() if v["dataset"] == ds and v["algo"] == "dt"]
    keys = list(runs[0]["by_target"].keys())
    xs = [normalized_score(env, float(k)) for k in keys]
    M = np.array([[np.mean(r["by_target"][k]["norm"]) for k in keys] for r in runs])
    ax.errorbar(xs, M.mean(0), M.std(0, ddof=1) if len(runs) > 1 else None, marker="o", capsize=3, label="DT")
    for algo, c in (("bc", "tab:red"), ("pbc", "tab:green")):
        v = per_seed(ds, algo); ax.axhline(np.mean(v), color=c, ls="--", label={"bc": "BC", "pbc": "%BC"}[algo])
    ax.axline((0, 0), slope=1, color="gray", ls=":", lw=1, label="score = target")
    ax.axvline(stats[ds][1], color="k", lw=0.6, alpha=0.5)
    ax.set_title(ds, fontsize=10); ax.set_xlabel("target return (normalized)"); ax.set_ylabel("achieved (normalized)")
    ax.set_xlim(min(xs) - 5, max(xs) + 5)
axs.flat[0].legend(fontsize=7)
for ax in list(axs.flat)[n:]: ax.axis("off")
fig.tight_layout(); fig.savefig("results/target_sweep.png", dpi=110)

fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
for f in sorted(glob.glob("logs/hopper-medium_*_s*.csv")):
    d = np.genfromtxt(f, delimiter=",", names=True); algo = f.split("_")[1]
    axs[0 if algo == "dt" else 1].plot(d["step"], d["loss"], lw=0.8, label=os.path.basename(f)[14:-4])
for ax, t in zip(axs, ("DT action MSE (hopper-medium)", "BC / %BC action MSE (hopper-medium)")):
    ax.set_title(t, fontsize=9); ax.set_yscale("log"); ax.set_xlabel("step"); ax.legend(fontsize=6)
fig.tight_layout(); fig.savefig("results/loss.png", dpi=100)
