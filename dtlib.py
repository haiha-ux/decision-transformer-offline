"""Data loading, Decision Transformer, BC models and evaluation. Everything here is plain PyTorch/numpy."""
import os, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MINARI_DATASETS_PATH", os.path.join(ROOT, "data"))

# D4RL reference returns (random policy, expert policy) used by d4rl.get_normalized_score;
# values from the D4RL repo (d4rl/infos.py, REF_MIN_SCORE / REF_MAX_SCORE).
REF = {"hopper": (-20.272305, 3234.3), "halfcheetah": (-280.178953, 12135.0), "walker2d": (1.629008, 4592.3)}
ENV_ID = {"hopper": "Hopper-v5", "halfcheetah": "HalfCheetah-v5", "walker2d": "Walker2d-v5"}
RTG_SCALE = 1000.0
MAX_T = 1000


def normalized_score(env, ret):
    lo, hi = REF[env]
    return 100.0 * (ret - lo) / (hi - lo)


def returns_to_go(rewards):
    """Undiscounted suffix sums: rtg[t] = sum_{t'>=t} r[t']."""
    return np.cumsum(np.asarray(rewards, dtype=np.float64)[::-1])[::-1].copy()


def parse_name(name):  # "hopper-medium" -> ("hopper", "medium")
    env, kind = name.split("-", 1)
    return env, kind


def load_trajs(name):
    """List of dicts(obs, act, rew), obs aligned with act (final next-obs dropped). Cached in data/cache."""
    cache = os.path.join(ROOT, "data", "cache", f"{name}.npz")
    if not os.path.exists(cache):
        import minari
        env, kind = parse_name(name)
        kinds = ["medium", "expert"] if kind == "medium-expert" else [kind]
        O, A, R, L = [], [], [], []
        for k in kinds:
            ds = minari.load_dataset(f"mujoco/{env}/{k}-v0", download=True)
            for ep in ds.iterate_episodes():
                O.append(ep.observations[:-1].astype(np.float32)); A.append(ep.actions.astype(np.float32))
                R.append(ep.rewards.astype(np.float32)); L.append(len(ep.rewards))
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, obs=np.concatenate(O), act=np.concatenate(A), rew=np.concatenate(R), lens=np.array(L))
    z = dict(np.load(cache))  # read each array once (NpzFile re-reads on every access)
    ends = np.cumsum(z["lens"]); starts = ends - z["lens"]
    return [dict(obs=z["obs"][s:e], act=z["act"][s:e], rew=z["rew"][s:e].copy()) for s, e in zip(starts, ends)]


def top_pct_trajs(trajs, pct=0.10):
    """Highest-return trajectories until they cover >= pct of all timesteps (the DT paper's %BC selection)."""
    order = np.argsort([-t["rew"].sum() for t in trajs])
    total, acc, keep = sum(len(t["rew"]) for t in trajs), 0, []
    for i in order:
        keep.append(trajs[i]); acc += len(trajs[i]["rew"])
        if acc >= pct * total: break
    return keep


class Batcher:
    """Samples K-step windows. Trajectory chosen proportional to its length, start uniform, left-padded like the DT reference code."""
    def __init__(self, trajs, K, device):
        self.K, self.dev, self.trajs = K, device, trajs
        allobs = np.concatenate([t["obs"] for t in trajs])
        self.mean, self.std = allobs.mean(0), allobs.std(0) + 1e-6
        lens = np.array([len(t["rew"]) for t in trajs]); self.p = lens / lens.sum()
        self.rtg = [returns_to_go(t["rew"]) for t in trajs]
        self.od, self.ad = trajs[0]["obs"].shape[1], trajs[0]["act"].shape[1]

    def sample(self, B, rng):
        """Vectorised window gather: pick trajectory (prob ~ length), start uniform, take up to K steps, left-pad."""
        K = self.K
        if not hasattr(self, "_g"):
            lens = np.array([len(t["rew"]) for t in self.trajs])
            self._g = dict(lens=lens, st=np.cumsum(lens) - lens, obs=((np.concatenate([t["obs"] for t in self.trajs]) - self.mean) / self.std).astype(np.float32),
                           act=np.concatenate([t["act"] for t in self.trajs]), rtg=(np.concatenate(self.rtg) / RTG_SCALE).astype(np.float32))
        g_ = self._g
        i = rng.choice(len(self.trajs), B, p=self.p); n = g_["lens"][i]; si = (rng.random(B) * n).astype(np.int64)
        l = np.minimum(K, n - si); pos = np.arange(K)[None] - (K - l)[:, None]; m = (pos >= 0)
        gi = np.where(m, g_["st"][i][:, None] + si[:, None] + pos, 0)
        f = lambda x: torch.as_tensor(x, device=self.dev)
        mk = m.astype(np.float32)
        return (f(g_["rtg"][gi][..., None] * mk[..., None]), f(g_["obs"][gi] * mk[..., None]), f(g_["act"][gi] * mk[..., None]),
                f(np.where(m, si[:, None] + pos, 0)), f(mk))

    def sample_flat(self, B, rng):
        """(state, action) pairs for BC, uniform over timesteps of the given trajectories."""
        if not hasattr(self, "_flat"):
            self._flat = (np.concatenate([(t["obs"] - self.mean) / self.std for t in self.trajs]), np.concatenate([t["act"] for t in self.trajs]))
        idx = rng.integers(0, len(self._flat[0]), B)
        return torch.as_tensor(self._flat[0][idx], device=self.dev), torch.as_tensor(self._flat[1][idx], device=self.dev)


def causal_key_mask(mask_tokens):
    """Boolean attention mask (B,L,L): query i may attend key j iff j<=i and key j is a real (non-pad) token. Diagonal always allowed so pad rows stay finite."""
    B, L = mask_tokens.shape
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=mask_tokens.device))
    m = causal[None] & mask_tokens[:, None, :].bool()
    idx = torch.arange(L, device=mask_tokens.device)
    return m | (idx[:, None] == idx[None, :])[None]  # (not torch.eye: ONNX EyeLike has no bool kernel)


class Block(nn.Module):
    def __init__(self, d, heads, drop):
        super().__init__()
        self.h = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(drop))
        self.drop = nn.Dropout(drop)

    def forward(self, x, allow):  # allow: (B,L,L) bool
        B, L, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / (D // self.h) ** 0.5
        att = att.masked_fill(~allow[:, None], float("-inf")).softmax(-1)
        x = x + self.drop(self.proj((self.drop(att) @ v).transpose(1, 2).reshape(B, L, D)))
        return x + self.mlp(self.ln2(x))


class DecisionTransformer(nn.Module):
    """Tokens per step: (return-to-go, state, action), interleaved in that order. The action for step t is predicted from the state token of step t,
    so it sees R_t, s_t and everything earlier but never a_t."""
    def __init__(self, od, ad, d=128, layers=3, heads=1, drop=0.1, max_t=MAX_T + 1):
        super().__init__()
        self.od, self.ad = od, ad
        self.er, self.es, self.ea = nn.Linear(1, d), nn.Linear(od, d), nn.Linear(ad, d)
        self.et = nn.Embedding(max_t, d)
        self.ln0, self.drop = nn.LayerNorm(d), nn.Dropout(drop)
        self.blocks = nn.ModuleList([Block(d, heads, drop) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, ad), nn.Tanh())

    def forward(self, rtg, s, a, ts, mask):
        B, K, _ = s.shape
        te = self.et(ts)
        tok = torch.stack([self.er(rtg) + te, self.es(s) + te, self.ea(a) + te], 2).reshape(B, 3 * K, -1)
        m3 = mask.repeat_interleave(3, 1)
        x = self.drop(self.ln0(tok)); allow = causal_key_mask(m3)
        for blk in self.blocks: x = blk(x, allow)
        x = self.lnf(x).reshape(B, K, 3, -1)
        return self.head(x[:, :, 1])  # state-token outputs -> actions


def make_mlp(od, ad, h=256):
    return nn.Sequential(nn.Linear(od, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, ad), nn.Tanh())


@torch.no_grad()
def eval_dt(model, env_name, mean, std, targets, n_eps, seed, device, K=20, render_first=None):
    """Roll out DT for every (target, episode) pair in lockstep. Returns {target: [episode returns]}."""
    import gymnasium as gym
    model.eval()
    od, ad = model.od, model.ad
    pairs = [(g, i) for g in targets for i in range(n_eps)]
    N = len(pairs)
    envs = [gym.make(ENV_ID[env_name]) for _ in range(N)]
    obs = np.stack([e.reset(seed=seed * 10000 + i)[0] for e, (g, i) in zip(envs, pairs)])
    S = torch.zeros(N, MAX_T + 1, od, device=device); A = torch.zeros(N, MAX_T + 1, ad, device=device)
    G = torch.zeros(N, MAX_T + 1, 1, device=device)
    G[:, 0, 0] = torch.tensor([g / RTG_SCALE for g, _ in pairs], device=device)
    mu, sd = torch.as_tensor(mean, device=device), torch.as_tensor(std, device=device)
    rets, alive, frames = np.zeros(N), np.ones(N, bool), []
    for t in range(MAX_T):
        S[:, t] = (torch.as_tensor(obs, dtype=torch.float32, device=device) - mu) / sd
        lo = max(0, t - K + 1); l = t - lo + 1
        s, a, g = [torch.zeros(N, K, x.shape[-1], device=device) for x in (S, A, G)]
        s[:, K - l:], a[:, K - l:], g[:, K - l:] = S[:, lo:t + 1], A[:, lo:t + 1], G[:, lo:t + 1]
        ts = torch.zeros(N, K, dtype=torch.long, device=device); ts[:, K - l:] = torch.arange(lo, t + 1, device=device)
        m = torch.zeros(N, K, device=device); m[:, K - l:] = 1
        act = model(g, s, a, ts, m)[:, -1]
        A[:, t] = act
        act_np = act.cpu().numpy()
        for j in np.where(alive)[0]:
            o, r, te, tr, _ = envs[j].step(act_np[j])
            obs[j] = o; rets[j] += r; G[j, t + 1, 0] = G[j, t, 0] - r / RTG_SCALE
            if te or tr: alive[j] = False
        if not alive.any(): break
    for e in envs: e.close()
    out = {g: [] for g in targets}
    for (g, _), r in zip(pairs, rets): out[g].append(float(r))
    return out


@torch.no_grad()
def eval_bc(net, env_name, mean, std, n_eps, seed, device):
    import gymnasium as gym
    net.eval()
    envs = [gym.make(ENV_ID[env_name]) for _ in range(n_eps)]
    obs = np.stack([e.reset(seed=seed * 10000 + i)[0] for i, e in enumerate(envs)])
    rets, alive = np.zeros(n_eps), np.ones(n_eps, bool)
    for t in range(MAX_T):
        x = torch.as_tensor((obs - mean) / std, dtype=torch.float32, device=device)
        act = net(x).cpu().numpy()
        for j in np.where(alive)[0]:
            o, r, te, tr, _ = envs[j].step(act[j]); obs[j] = o; rets[j] += r
            if te or tr: alive[j] = False
        if not alive.any(): break
    for e in envs: e.close()
    return [float(r) for r in rets]
