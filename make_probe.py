"""Record the first 100 steps of the seed-0 hopper-medium DT rollout (target = 1.0 x best dataset return, env seed 12345, as in make_demo.py)
-> docs/probe.json. The page replays these normalised states in the browser through hopper_dt.onnx with any target return."""
import json, numpy as np, torch, gymnasium as gym
from dtlib import *
torch.set_num_threads(2)
N, K = 100, 20
c = torch.load("ckpt/hopper-medium_dt_s0.pt", weights_only=False)
net = DecisionTransformer(c["od"], c["ad"]); net.load_state_dict(c["model"]); net.eval()
target = float(max(t["rew"].sum() for t in load_trajs("hopper-medium")))
env = gym.make(ENV_ID["hopper"]); o, _ = env.reset(seed=12345)
S, A, R = [], [], []
with torch.no_grad():
    for t in range(N):
        S.append(((o - c["mean"]) / c["std"]).astype(np.float32))
        lo = max(0, t - K + 1); l = t - lo + 1
        s, a, g = torch.zeros(1, K, c["od"]), torch.zeros(1, K, c["ad"]), torch.zeros(1, K, 1)
        s[0, K - l:] = torch.as_tensor(np.stack(S[lo:t + 1]))
        if t > lo: a[0, K - l:K - 1] = torch.as_tensor(np.stack(A[lo:t]))  # a_t itself stays 0, as in eval_dt
        g[0, K - l:, 0] = torch.as_tensor([(target - sum(R[:j])) / RTG_SCALE for j in range(lo, t + 1)], dtype=torch.float32)
        ts = torch.zeros(1, K, dtype=torch.long); ts[0, K - l:] = torch.arange(lo, t + 1)
        m = torch.zeros(1, K); m[0, K - l:] = 1
        act = net(g, s, a, ts, m)[0, -1].numpy(); A.append(act)
        o, r, te, tr, _ = env.step(act); R.append(float(r))
        assert not (te or tr)
r5 = lambda x: np.round(np.asarray(x, dtype=np.float64), 5).tolist()
json.dump(dict(K=K, target=target, rtg_scale=RTG_SCALE, states=r5(S), actions=r5(A), rewards=r5(R)), open("docs/probe.json", "w"), separators=(",", ":"))
print("steps", N, "return so far", round(sum(R), 1), "target", target)
