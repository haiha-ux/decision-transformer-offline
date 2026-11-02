"""Cheap checks on existing DT/BC checkpoints (no training): does the model use RTG, does eval-time RTG match training RTG,
is the eval window construction identical to the training one.  python diagnose.py [dataset [tag]]  -> results/diagnostics_<dataset>.json"""
import sys, json, glob
import numpy as np, torch
from dtlib import *

ds = sys.argv[1] if len(sys.argv) > 1 else "hopper-medium-expert"
tag = sys.argv[2] if len(sys.argv) > 2 else ""   # "ref" -> checkpoints <ds>_dtref_s*.pt
torch.set_num_threads(2); torch.manual_seed(0)
trajs = load_trajs(ds)
rets = np.array([t["rew"].sum() for t in trajs]); lens = np.array([len(t["rew"]) for t in trajs])
out = {"dataset": ds, "n_traj": len(trajs)}
# --- medium vs expert separation (cache = medium trajectories first, then expert) ---
# mixed cache boundary: first index where cumulative steps reaches medium dataset size is unknown offline, so use the return gap
order = np.argsort(rets)
hi_mask = rets > (np.percentile(rets, 50) if ds != "hopper-medium-expert" else 0.5 * (rets.min() + rets.max()))
if ds == "hopper-medium-expert":
    ncut = None
    cs = np.cumsum(lens); first_half = np.searchsorted(cs, cs[-1] / 2)  # medium and expert are 1M steps each
    out["cache_order_check"] = dict(medium_part_return_mean=float(rets[:first_half].mean()), expert_part_return_mean=float(rets[first_half:].mean()),
                                    medium_part_return_max=float(rets[:first_half].max()), expert_part_return_min=float(rets[first_half:].min()),
                                    medium_part_return_pctl=np.percentile(rets[:first_half], [5, 50, 95]).tolist(),
                                    expert_part_return_pctl=np.percentile(rets[first_half:], [5, 50, 95]).tolist(),
                                    n_medium=int(first_half), n_expert=int(len(rets) - first_half), steps_medium=int(cs[first_half - 1]), steps_expert=int(cs[-1] - cs[first_half - 1]))
    is_exp = np.zeros(len(trajs), bool); is_exp[first_half:] = True
else:
    is_exp = hi_mask

bt = Batcher(trajs, 20, torch.device("cpu"))
# --- RTG at a given timestep in training data vs eval targets ---
mx = float(rets.max()); out["max_return"] = mx
out["eval_targets"] = [round(f * mx, 1) for f in [0.4, 0.6, 0.8, 1.0, 1.2]]
for t_ in (0, 100, 300):
    v = np.array([r[t_] for r in bt.rtg if len(r) > t_])
    out[f"train_rtg_at_t{t_}_pctl_1_25_50_75_99"] = np.percentile(v, [1, 25, 50, 75, 99]).round(1).tolist()
out["train_rtg_at_t0_by_group"] = {"medium": np.percentile([bt.rtg[i][0] for i in range(len(trajs)) if not is_exp[i]], [5, 50, 95]).round(1).tolist(),
                                   "expert": np.percentile([bt.rtg[i][0] for i in range(len(trajs)) if is_exp[i]], [5, 50, 95]).round(1).tolist()}
# --- model use of RTG, per checkpoint ---
rng = np.random.default_rng(123); N = 3000
g, s, a, ts, m = bt.sample(N, rng)           # fixed held-out-style windows (same sampler as training, different rng)
# which group each window came from: re-derive by sampling index with the same rng stream is awkward -> classify by RTG of first real token
first = m.argmax(1)
g0 = g[torch.arange(N), first, 0].numpy() * RTG_SCALE
rtg_cut = 0.5 * (np.percentile(rets, 5) + np.percentile(rets, 95)) if ds == "hopper-medium-expert" else np.median(rets)
# window group by its RTG at first token, only meaningful at t~0; use windows with ts at first real token < 60 as "start" windows
t0 = ts[torch.arange(N), first].numpy()
grp_exp = g0 > np.median(rets[is_exp]) * 0.0 + 0.5 * (rets[~is_exp].mean() + rets[is_exp].mean()) - 0.0
def mse(model, G):
    with torch.no_grad(): p = model(G, s, a, ts, m)
    return float((((p - a) ** 2).mean(-1) * m).sum() / m.sum()), p
def mid_pred_diff(model, shift):
    with torch.no_grad():
        p0 = model(g, s, a, ts, m)
        p1 = model(g + shift * m[..., None], s, a, ts, m)
    d = ((p1 - p0).abs().mean(-1) * m).sum(1) / m.sum(1)
    return d
res = {}
for path in sorted(glob.glob(f"ckpt/{ds}_dt{tag}_s*.pt")):
    c = torch.load(path, weights_only=False)
    assert np.allclose(c["mean"], bt.mean, atol=1e-4) and np.allclose(c["std"], bt.std, atol=1e-4), "ckpt normalisation differs from Batcher"
    net = DecisionTransformer(c["od"], c["ad"]); net.load_state_dict(c["model"]); net.eval()
    r = {}
    r["mse_true_rtg"] = mse(net, g)[0]
    perm = torch.randperm(N); r["mse_shuffled_rtg_across_windows"] = mse(net, g[perm])[0]
    r["mse_const_rtg_mean"] = mse(net, torch.full_like(g, float(np.mean(rets)) / RTG_SCALE) * m[..., None])[0]
    r["mse_zero_rtg"] = mse(net, torch.zeros_like(g))[0]
    for lo_, hi_ in [(1.0, 3.5)]:
        # constant low vs high return-to-go over the whole window, split by expert/medium windows (window group = trajectory group, via RTG at window start)
        pl = net(torch.full_like(g, lo_) * m[..., None], s, a, ts, m); ph = net(torch.full_like(g, hi_) * m[..., None], s, a, ts, m)
        d = ((ph - pl).abs().mean(-1) * m).sum(1) / m.sum(1)
        r["mean_abs_action_change_const_rtg_1000_to_3500_all"] = float(d.mean())
        r["action_abs_scale"] = float(a.abs().mean())
    r["mean_abs_action_change_rtg_shift_plus1000_all"] = float(mid_pred_diff(net, 1.0).mean())
    res[path] = r
out["dt"] = res
# --- BC / DT offline mse on the same windows (state-only), same state/action pairs ---
bcm = {}
for path in sorted(glob.glob(f"ckpt/{ds}_bc_s*.pt")) + sorted(glob.glob(f"ckpt/{ds}_pbc_s*.pt")):
    c = torch.load(path, weights_only=False)
    net = make_mlp(c["od"], c["ad"]); net.load_state_dict(c["model"]); net.eval()
    with torch.no_grad(): p = net(s)
    bcm[path] = float((((p - a) ** 2).mean(-1) * m).sum() / m.sum())
out["bc_mse_same_windows"] = bcm
out["mean_action_mse_of_zero_pred"] = float(((a ** 2).mean(-1) * m).sum() / m.sum())
# --- per-group (by trajectory return, medium vs expert) on-trajectory RTG sensitivity ---
# build windows per trajectory group explicitly
def windows_from(idx_list, n_per=1500):
    rr = np.random.default_rng(7); G_, S_, A_, T_, M_ = [], [], [], [], []
    for _ in range(n_per):
        i = idx_list[rr.integers(len(idx_list))]; n = lens[i]; si = int(rr.integers(0, max(1, n - 20)))   # full windows only
        sl = slice(si, si + 20)
        G_.append(bt.rtg[i][sl] / RTG_SCALE); S_.append((trajs[i]["obs"][sl] - bt.mean) / bt.std); A_.append(trajs[i]["act"][sl]); T_.append(np.arange(si, si + 20))
    f = lambda x, dt=torch.float32: torch.as_tensor(np.array(x), dtype=dt)
    return f(G_)[..., None], f(S_), f(A_), f(T_, torch.long), torch.ones(n_per, 20)
grp = {}
for name, idx in [("medium", np.where(~is_exp)[0]), ("expert", np.where(is_exp)[0])]:
    G_, S_, A_, T_, M_ = windows_from(idx)
    gg = {}
    for path in sorted(glob.glob(f"ckpt/{ds}_dt{tag}_s*.pt")):
        c = torch.load(path, weights_only=False)
        net = DecisionTransformer(c["od"], c["ad"]); net.load_state_dict(c["model"]); net.eval()
        with torch.no_grad():
            pt = net(G_, S_, A_, T_, M_)
            plo = net(torch.full_like(G_, 1.0), S_, A_, T_, M_); phi = net(torch.full_like(G_, 3.5), S_, A_, T_, M_)
            psh = net(G_[torch.randperm(len(G_))], S_, A_, T_, M_)
        gg[path] = dict(mse_true=float(((pt - A_) ** 2).mean()), mse_shuffled=float(((psh - A_) ** 2).mean()), mse_rtg1000=float(((plo - A_) ** 2).mean()),
                        mse_rtg3500=float(((phi - A_) ** 2).mean()), mean_abs_change_1000_vs_3500=float((phi - plo).abs().mean()),
                        mean_abs_change_true_vs_shuffled=float((pt - psh).abs().mean()), true_rtg_mean_x1000=float(G_.mean() * 1000))
    grp[name] = gg
    bg = {}
    for path in sorted(glob.glob(f"ckpt/{ds}_bc_s*.pt")):
        c = torch.load(path, weights_only=False)
        net = make_mlp(c["od"], c["ad"]); net.load_state_dict(c["model"]); net.eval()
        with torch.no_grad(): bg[path] = float(((net(S_) - A_) ** 2).mean())
    grp[name + "_bc_mse"] = bg
out["by_group_full_windows"] = grp
json.dump(out, open(f"results/diagnostics_{ds}{tag}.json", "w"), indent=1)
print(json.dumps(out, indent=1))
