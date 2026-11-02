"""Export the seed-0 hopper-medium DT to ONNX (fixed window K=20) and check parity with PyTorch via onnxruntime."""
import numpy as np, torch, onnxruntime as ort
from dtlib import *
c = torch.load("ckpt/hopper-medium_dt_s0.pt", weights_only=False)
net = DecisionTransformer(c["od"], c["ad"]); net.load_state_dict(c["model"]); net.eval()
K = 20
ex = (torch.randn(1, K, 1), torch.randn(1, K, c["od"]), torch.randn(1, K, c["ad"]), torch.arange(K)[None], torch.ones(1, K))
torch.onnx.export(net, ex, "docs/hopper_dt.onnx", dynamo=False, opset_version=17,
                  input_names=["rtg", "states", "actions", "timesteps", "mask"], output_names=["actions_pred"])
sess = ort.InferenceSession("docs/hopper_dt.onnx")
worst = 0
for _ in range(20):
    x = (torch.randn(1, K, 1), torch.randn(1, K, c["od"]), torch.randn(1, K, c["ad"]), torch.randint(0, 1000, (1, K)), (torch.rand(1, K) > 0.3).float())
    x[4][0, -1] = 1
    ref = net(*x).detach().numpy()
    out = sess.run(None, dict(zip(["rtg", "states", "actions", "timesteps", "mask"], [t.numpy() for t in x])))[0]
    worst = max(worst, float(np.abs(ref - out).max()))
print("max abs diff torch vs onnxruntime over 20 random inputs:", worst)
np.savez("docs/hopper_dt_norm.npz", mean=c["mean"], std=c["std"])
