"""python eval.py [--runs glob]   -> results/eval.json (appends/overwrites per run). 10 episodes per (run, target), fresh env seeds."""
import argparse, glob, json, os
import numpy as np, torch
from dtlib import *

p = argparse.ArgumentParser()
p.add_argument("--glob", default="ckpt/*.pt"); p.add_argument("--n-eps", type=int, default=10)
p.add_argument("--out", default="results/eval.json")
a = p.parse_args()
torch.set_num_threads(2)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
res = json.load(open(a.out)) if os.path.exists(a.out) else {}
FRACS = [0.4, 0.6, 0.8, 1.0, 1.2]
for path in sorted(glob.glob(a.glob)):
    c = torch.load(path, weights_only=False)
    name = os.path.basename(path)[:-3]
    if name in res: continue
    env, _ = parse_name(c["dataset"])
    if c["algo"] == "dt":
        net = DecisionTransformer(c["od"], c["ad"]).to(dev); net.load_state_dict(c["model"])
        mx = max(t["rew"].sum() for t in load_trajs(c["dataset"]))  # target scale: best return in the dataset
        targets = [float(f * mx) for f in FRACS]
        out = eval_dt(net, env, c["mean"], c["std"], targets, a.n_eps, c["seed"] + 1, dev)
        res[name] = dict(algo="dt", dataset=c["dataset"], seed=c["seed"], max_dataset_return=float(mx),
                         by_target={str(round(g, 1)): dict(frac=f, returns=r, norm=[normalized_score(env, x) for x in r])
                                    for f, (g, r) in zip(FRACS, out.items())})
    else:
        net = make_mlp(c["od"], c["ad"]).to(dev); net.load_state_dict(c["model"])
        r = eval_bc(net, env, c["mean"], c["std"], a.n_eps, c["seed"] + 1, dev)
        res[name] = dict(algo=c["algo"], dataset=c["dataset"], seed=c["seed"], returns=r, norm=[normalized_score(env, x) for x in r])
    os.makedirs("results", exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
    v = res[name]
    print(name, {k: round(float(np.mean(x["norm"])), 1) for k, x in v["by_target"].items()} if "by_target" in v else round(float(np.mean(v["norm"])), 1), flush=True)
