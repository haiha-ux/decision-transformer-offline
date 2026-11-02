"""Render seed-0 DT rollouts. python make_demo.py  -> docs/hopper/target_*.mp4 (+ docs/manifest.json), docs/<Env>.gif/.mp4. Needs MUJOCO_GL=egl and ffmpeg."""
import os, sys, json, subprocess
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch, gymnasium as gym
from PIL import Image
from dtlib import *
torch.set_num_threads(2)
SKIP = 4

@torch.no_grad()
def rollout(path, frac, seed=12345, K=20, max_frames=None):
    c = torch.load(path, weights_only=False); env_name, _ = parse_name(c["dataset"])
    net = DecisionTransformer(c["od"], c["ad"]); net.load_state_dict(c["model"]); net.eval()
    mx = max(t["rew"].sum() for t in load_trajs(c["dataset"])); target = float(frac * mx)
    env = gym.make(ENV_ID[env_name], render_mode="rgb_array", width=384, height=288)
    o, _ = env.reset(seed=seed)
    od, ad = c["od"], c["ad"]
    S, A, G = torch.zeros(1, MAX_T + 1, od), torch.zeros(1, MAX_T + 1, ad), torch.zeros(1, MAX_T + 1, 1)
    G[0, 0, 0] = target / RTG_SCALE; mu, sd = torch.as_tensor(c["mean"]), torch.as_tensor(c["std"])
    frames, R, t = [], 0.0, 0
    while t < MAX_T:
        S[0, t] = (torch.as_tensor(o, dtype=torch.float32) - mu) / sd
        lo = max(0, t - K + 1); l = t - lo + 1
        s, a, g = torch.zeros(1, K, od), torch.zeros(1, K, ad), torch.zeros(1, K, 1)
        s[:, K - l:], a[:, K - l:], g[:, K - l:] = S[:, lo:t + 1], A[:, lo:t + 1], G[:, lo:t + 1]
        ts = torch.zeros(1, K, dtype=torch.long); ts[:, K - l:] = torch.arange(lo, t + 1)
        m = torch.zeros(1, K); m[:, K - l:] = 1
        act = net(g, s, a, ts, m)[0, -1]; A[0, t] = act
        o, r, te, tr, _ = env.step(act.numpy()); R += r; G[0, t + 1, 0] = G[0, t, 0] - r / RTG_SCALE; t += 1
        if t % SKIP == 0 and (max_frames is None or len(frames) < max_frames): frames.append(env.render())
        if te or tr or (max_frames and len(frames) >= max_frames): break
    fps = round(1 / (env.unwrapped.dt * SKIP))
    return dict(env=env_name, frames=frames, fps=fps, ret=R, steps=t, terminated=bool(te), target=target, frac=frac,
                norm=normalized_score(env_name, R), max_frames=max_frames)

def write_mp4(frames, fps, out):
    h, w = frames[0].shape[:2]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "30", out], input=np.stack(frames).tobytes(), check=True)

def write_gif(frames, fps, out):
    small = [Image.fromarray(f).resize((256, 192)) for f in frames]
    small[0].save(out, save_all=True, append_images=small[1:], duration=int(1000 / fps), loop=0)

if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "all"  # "page" = hopper slider videos, "gifs" = per-env clips
    os.makedirs("docs/hopper", exist_ok=True)
    man = []
    for frac in (0.4, 0.6, 1.0, 1.2) if part in ("page", "all") else ():  # hopper-medium page: every target, one fixed env seed, no selection
        r = rollout("ckpt/hopper-medium_dt_s0.pt", frac)
        fn = f"hopper/target_{int(round(frac * 100))}.mp4"; write_mp4(r["frames"], r["fps"], "docs/" + fn)
        man.append(dict(file=fn, frac=frac, target=round(r["target"], 1), ret=round(r["ret"], 1), norm=round(r["norm"], 1), steps=r["steps"], terminated=r["terminated"]))
        print(man[-1], flush=True)
    if man:
        json.dump(man, open("docs/manifest.json", "w"), indent=1)
        html = open("docs/index.html").read()
        i = html.index("const M = ") + 10; j = html.index(";\n", i)
        open("docs/index.html", "w").write(html[:i] + json.dumps(man) + html[j:])  # embed manifest so the page works from file://
    for ds in ("hopper-medium", "halfcheetah-medium", "walker2d-medium") if part in ("gifs", "all") else ():
        r = rollout(f"ckpt/{ds}_dt_s0.pt", 1.0, max_frames=200)
        n = ENV_ID[parse_name(ds)[0]].split("-")[0]
        write_mp4(r["frames"], r["fps"], f"docs/{n}.mp4"); write_gif(r["frames"], r["fps"], f"docs/{n}.gif")
        print(n, "return so far", round(r["ret"], 1), "steps", r["steps"], "terminated" if r["terminated"] else "cut at frame cap",
              [os.path.getsize(f"docs/{n}.{e}") // 1024 for e in ("gif", "mp4")], "KB", flush=True)
    os._exit(0)
