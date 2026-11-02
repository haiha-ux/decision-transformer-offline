// Replays a recorded state sequence through the ONNX Decision Transformer with a chosen target return.
// Same windowing as eval_dt in dtlib.py: last K steps, left-padded with zeros, mask=0 on padding, a_t itself not given (zeros).
async function runProbe(ort, sess, P, target) {
  const K = P.K, od = P.states[0].length, ad = P.actions[0].length, out = [];
  for (let t = 0; t < P.states.length; t++) {
    const lo = Math.max(0, t - K + 1), l = t - lo + 1, p = K - l;
    const rtg = new Float32Array(K), st = new Float32Array(K * od), ac = new Float32Array(K * ad);
    const ts = new BigInt64Array(K), mask = new Float32Array(K);
    let spent = 0;
    for (let j = 0; j < lo; j++) spent += P.rewards[j];
    for (let j = lo; j <= t; j++) {
      const i = p + j - lo;
      rtg[i] = (target - spent) / P.rtg_scale; spent += P.rewards[j];
      st.set(P.states[j], i * od); if (j < t) ac.set(P.actions[j], i * ad);
      ts[i] = BigInt(j); mask[i] = 1;
    }
    const r = await sess.run({
      rtg: new ort.Tensor("float32", rtg, [1, K, 1]), states: new ort.Tensor("float32", st, [1, K, od]),
      actions: new ort.Tensor("float32", ac, [1, K, ad]), timesteps: new ort.Tensor("int64", ts, [1, K]), mask: new ort.Tensor("float32", mask, [1, K]),
    });
    out.push(Array.from(r.actions_pred.data.slice((K - 1) * ad, K * ad)));
  }
  return out;
}
if (typeof module !== "undefined") module.exports = { runProbe };
