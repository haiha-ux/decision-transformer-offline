"""python train.py --algo {dt,bc,pbc} --dataset hopper-medium --seed 0 [--steps N] [--lr X] [--tag _name]"""
import argparse, os, time, json, csv, math
import numpy as np, torch, torch.nn.functional as F
import dtlib
from dtlib import *

p = argparse.ArgumentParser()
p.add_argument("--algo", required=True, choices=["dt", "bc", "pbc"])
p.add_argument("--dataset", required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--steps", type=int, default=None)
p.add_argument("--batch", type=int, default=None)
p.add_argument("--device", default=None)
p.add_argument("--lr", type=float, default=None)   # default: 3e-4 (dt), 1e-3 (bc/pbc)
p.add_argument("--tag", default="")                 # suffix for ckpt/log names, keeps side runs apart from the main grid
a = p.parse_args()
torch.set_num_threads(2)
dev = torch.device(a.device or ("cuda" if a.algo == "dt" and torch.cuda.is_available() else "cpu"))
if dev.type == "cuda": torch.cuda.set_per_process_memory_fraction(0.3)
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
steps = a.steps or (40000 if a.algo == "dt" else 10000)
B = a.batch or (64 if a.algo == "dt" else 256)

trajs = load_trajs(a.dataset)
if a.algo == "pbc": trajs = top_pct_trajs(trajs, 0.10)
bt = Batcher(trajs, 20, dev)
print(f"{a.dataset} {a.algo}: {len(trajs)} trajectories, {sum(len(t['rew']) for t in trajs)} steps", flush=True)
if a.algo == "dt":
    net = DecisionTransformer(bt.od, bt.ad).to(dev)
    lr, wd, warm = a.lr or 3e-4, 1e-4, 1000
else:
    net = make_mlp(bt.od, bt.ad).to(dev)
    lr, wd, warm = a.lr or 1e-3, 0.0, 200
opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(i, steps) / steps)) * 0.99 + 0.01))
name = f"{a.dataset}_{a.algo}{a.tag}_s{a.seed}"
os.makedirs("logs", exist_ok=True); os.makedirs("ckpt", exist_ok=True)
f = open(f"logs/{name}.csv", "w", newline=""); w = csv.writer(f); w.writerow(["step", "loss", "lr", "elapsed_s"])
t0, acc = time.time(), []
for i in range(1, steps + 1):
    net.train()
    if a.algo == "dt":
        g, s, ac, ts, m = bt.sample(B, rng)
        pred = net(g, s, ac, ts, m)
        loss = (((pred - ac) ** 2).mean(-1) * m).sum() / m.sum()
    else:
        s, ac = bt.sample_flat(B, rng)
        loss = F.mse_loss(net(s), ac)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 0.25 if a.algo == "dt" else 1.0)
    opt.step(); sched.step(); acc.append(loss.item())
    if i % 500 == 0:
        w.writerow([i, np.mean(acc), sched.get_last_lr()[0], round(time.time() - t0, 1)]); f.flush()
        if i % 5000 == 0: print(i, round(float(np.mean(acc)), 5), round(time.time() - t0), "s", flush=True)
        acc = []
f.close()
torch.save(dict(model=net.state_dict(), mean=bt.mean, std=bt.std, algo=a.algo, dataset=a.dataset, seed=a.seed, steps=steps, lr=lr, od=bt.od, ad=bt.ad), f"ckpt/{name}.pt")
json.dump(dict(train_seconds=round(time.time() - t0, 1), steps=steps, batch=B, n_traj=len(trajs)), open(f"logs/{name}.time.json", "w"))
print("done", name, round(time.time() - t0), "s")
