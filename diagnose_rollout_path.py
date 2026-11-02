"""Does eval_dt build the same model input as the training batcher? Replays dataset trajectories through the real eval_dt loop
(env replaced by a replay env, model replaced by a recorder that returns the dataset action), then feeds the recorded eval windows and
the equivalent training-style windows to the real checkpoint. python diagnose_rollout_path.py -> results/diagnostics_rollout_path.json"""
import json, sys
import numpy as np, torch, gymnasium
from dtlib import *

ds = "hopper-medium-expert"; torch.set_num_threads(2)
trajs = load_trajs(ds); bt = Batcher(trajs, 20, torch.device("cpu"))
c = torch.load(f"ckpt/{ds}_dt_s0.pt", weights_only=False)
net = DecisionTransformer(c["od"], c["ad"]); net.load_state_dict(c["model"]); net.eval()
assert np.allclose(c["mean"], bt.mean, atol=1e-4) and np.allclose(c["std"], bt.std, atol=1e-4)
sel = [i for i, t in enumerate(trajs) if len(t["rew"]) >= 400][::97][:16]
TS = (0, 3, 18, 19, 20, 60, 300)

class Replay:
    n = 0
    def __init__(self): self.i = sel[Replay.n]; Replay.n += 1; self.t = 0
    def reset(self, seed=None): self.t = 0; return trajs[self.i]["obs"][0].copy(), {}
    def step(self, a):
        T = trajs[self.i]; self.t += 1; r = float(T["rew"][self.t - 1]); last = self.t >= len(T["rew"])
        return (T["obs"][min(self.t, len(T["obs"]) - 1)].copy(), r, last, False, {})
    def close(self): pass

class Rec:
    od, ad = c["od"], c["ad"]; calls = 0; rec = {}
    def eval(self): pass
    def __call__(self, g, s, a, ts, m):
        t = Rec.calls; Rec.calls += 1
        if t in TS: Rec.rec[t] = [x.clone() for x in (g, s, a, ts, m)]
        out = torch.zeros(g.shape[0], g.shape[1], self.ad)
        for j, i in enumerate(sel): out[j, -1] = torch.as_tensor(trajs[i]["act"][min(t, len(trajs[i]["act"]) - 1)])
        return out
Replay.n = 0
orig = gymnasium.make; gymnasium.make = lambda *a, **k: Replay()
targets = [float(bt.rtg[i][0]) for i in sel]
eval_dt(Rec(), "hopper", c["mean"], c["std"], targets, 1, 0, torch.device("cpu"))
gymnasium.make = orig

rows = []
for t in TS:
    g, s, a, ts, m = Rec.rec[t]
    with torch.no_grad(): pe = net(g, s, a, ts, m)[:, -1]
    # training-style window: start at max(0,t-19)... full K tokens, take output at the position of step t
    G_, S_, A_, T_ = [], [], [], []
    for i in sel:
        st = max(0, t - 19); sl = slice(st, st + 20); n = len(trajs[i]["rew"])
        assert st + 20 <= n
        G_.append(bt.rtg[i][sl] / RTG_SCALE); S_.append((trajs[i]["obs"][sl] - bt.mean) / bt.std); A_.append(trajs[i]["act"][sl]); T_.append(np.arange(st, st + 20))
    f = lambda x, dt=torch.float32: torch.as_tensor(np.array(x), dtype=dt)
    pos = min(t, 19)
    with torch.no_grad(): pt = net(f(G_)[..., None], f(S_), f(A_), f(T_, torch.long), torch.ones(len(sel), 20))[:, pos]
    rows.append(dict(t=t, window_len=int(m[0].sum()), max_abs_action_diff_eval_vs_train=float((pe - pt).abs().max()),
                     max_abs_rtg_diff=float((g[:, -1, 0] - f(G_)[:, pos]).abs().max()), max_abs_state_diff=float((s[:, -1] - f(S_)[:, pos]).abs().max())))
    print(rows[-1])
json.dump(dict(dataset=ds, n_traj_checked=len(sel), checkpoint="ckpt/hopper-medium-expert_dt_s0.pt", rows=rows), open("results/diagnostics_rollout_path.json", "w"), indent=1)
