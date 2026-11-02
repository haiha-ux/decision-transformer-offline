import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from dtlib import returns_to_go, normalized_score, causal_key_mask, DecisionTransformer, top_pct_trajs, REF


def test_returns_to_go_hand_example():
    assert returns_to_go([1, 2, 3, 4]).tolist() == [10, 9, 7, 4]
    assert returns_to_go([5]).tolist() == [5]
    assert returns_to_go([1, -1, 1]).tolist() == [1, 0, 1]


def test_normalized_score_formula():
    for env, (lo, hi) in REF.items():
        assert normalized_score(env, lo) == 0 and abs(normalized_score(env, hi) - 100) < 1e-9
    assert abs(normalized_score("hopper", 1000.0) - 100 * 1020.272305 / 3254.572305) < 1e-9


def test_mask_is_causal_and_hides_padding():
    m = causal_key_mask(torch.tensor([[0., 0., 1., 1.]]))[0]
    assert not m[0, 1] and not m[2, 3]            # no looking forward
    assert m[3, 2] and m[3, 3]                     # real tokens see earlier real tokens
    assert not m[2, 0] and not m[3, 1]             # real tokens never see padding
    assert m[0, 0] and m[1, 1]                     # pad rows stay finite


def test_model_outputs_do_not_depend_on_future_tokens():
    torch.manual_seed(0)
    K, od, ad = 6, 5, 2
    net = DecisionTransformer(od, ad, d=32, layers=2, drop=0.0).eval()
    g, s, a = torch.randn(1, K, 1), torch.randn(1, K, od), torch.randn(1, K, ad)
    ts, mk = torch.arange(K)[None], torch.ones(1, K)
    base = net(g, s, a, ts, mk)
    for t in range(K):  # perturb everything from step t on; outputs before step t must not move
        g2, s2, a2 = g.clone(), s.clone(), a.clone()
        s2[:, t:] += 1; g2[:, t:] += 1; a2[:, t:] += 1
        out = net(g2, s2, a2, ts, mk)
        assert torch.allclose(out[:, :t], base[:, :t], atol=1e-6)
    # action a_t must not leak into its own prediction (predicted from the state token)
    a3 = a.clone(); a3[:, 3] += 5
    assert torch.allclose(net(g, s, a3, ts, mk)[:, :4], base[:, :4], atol=1e-6)
    assert not torch.allclose(net(g, s, a3, ts, mk)[:, 4:], base[:, 4:], atol=1e-6)


def test_padding_does_not_change_real_outputs():
    torch.manual_seed(1)
    net = DecisionTransformer(4, 2, d=32, layers=2, drop=0.0).eval()
    g, s, a = torch.randn(1, 3, 1), torch.randn(1, 3, 4), torch.randn(1, 3, 2); ts = torch.arange(3)[None]
    ref = net(g, s, a, ts, torch.ones(1, 3))
    pad = lambda x: torch.cat([torch.randn(1, 2, *x.shape[2:]), x], 1)
    out = net(pad(g), pad(s), pad(a), torch.cat([torch.zeros(1, 2, dtype=torch.long), ts], 1), torch.tensor([[0., 0, 1, 1, 1]]))
    assert torch.allclose(out[:, 2:], ref, atol=1e-5)


def test_top_pct_selection():
    trajs = [dict(rew=np.full(10, r, np.float32)) for r in (1, 5, 3, 9, 2, 0, 4, 7, 6, 8)]
    keep = top_pct_trajs(trajs, 0.2)
    assert [t["rew"][0] for t in keep] == [9, 8]
