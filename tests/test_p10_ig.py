import numpy as np
import torch

from cli_p5 import GRUClassifier
from app.explain.ig import IGConfig, integrated_gradients, topk_sparsify


def test_ig_shape_and_topk():
    W, F = 144, 6
    # Toy model
    net = GRUClassifier(input_dim=F, hidden=8)
    win = np.random.randn(W, F).astype(np.float32)
    cfg = IGConfig(steps=8, baseline="zeros", target=1, topk=5)
    attr = integrated_gradients(net, win, cfg)
    assert attr.shape == (W, F)
    top = topk_sparsify(attr, k=cfg.topk, feature_names=[f"f{i}" for i in range(F)])
    assert len(top) == cfg.topk
    # Deterministic across calls (model in eval and same win)
    attr2 = integrated_gradients(net, win, cfg)
    assert np.allclose(attr, attr2)

