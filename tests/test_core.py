import numpy as np
import torch

from damagelens.align import apply_shift, estimate_shift, register_orb, register_phase
from damagelens.augment import misregister
from damagelens.baselines import BASELINES, dnbr
from damagelens.data import tiles
from damagelens.metrics import binary_scores, building_confusion, xview2_scores
from damagelens.models import build_unet, predict
from damagelens.summarize import grid_summary


def scene(seed=0, size=256):
    rng = np.random.default_rng(seed)
    img = rng.normal(110, 18, (size, size, 3)).clip(0, 255).astype(np.uint8)
    for _ in range(40):
        y, x = rng.integers(0, size - 30, 2); h, w = rng.integers(8, 30, 2)
        img[y:y + h, x:x + w] = rng.integers(40, 255, 3)
    return img


def test_phase_correlation_recovers_shift():
    a = scene()
    b = apply_shift(a, np.array([6.0, -4.0]))
    assert np.allclose(estimate_shift(a, b), [-6, 4], atol=0.5)
    aligned, sh, ok = register_phase(a, b)
    assert ok and np.abs(aligned.astype(int) - a.astype(int))[20:-20, 20:-20].mean() < 5


def test_orb_registration_recovers_shift():
    a = scene(1)
    b = apply_shift(a, np.array([5.0, 3.0]))
    aligned, model, n, ok = register_orb(a, b)
    assert ok and n >= 12
    assert np.allclose(model.translation, [-3, -5], atol=1.0)


def test_baselines_flag_the_changed_block():
    a = scene(2); b = a.copy(); b[100:160, 100:160] = [250, 250, 250]
    gt = np.zeros(a.shape[:2], bool); gt[100:160, 100:160] = True
    for name, fn in BASELINES.items():
        s = fn(a, b)
        assert s.shape == gt.shape
        assert s[gt].mean() > s[~gt].mean(), name


def test_dnbr_sign():
    assert dnbr(np.array(0.4), np.array(0.1), np.array(0.1), np.array(0.3)) > 0.66


def test_tiles_cover_image():
    cover = np.zeros((1000, 700), int)
    for y, x in tiles(1000, 700, 512):
        cover[y:y + 512, x:x + 512] += 1
    assert cover.min() >= 1


def test_metrics():
    assert binary_scores([1, 1, 0, 0], [1, 0, 1, 0])["f1"] == 0.5
    gt = np.zeros((20, 20), np.uint8); gt[2:6, 2:6] = 4; gt[10:14, 10:14] = 1
    cm = building_confusion(gt, gt)
    assert cm[4, 4] == 1 and cm[1, 1] == 1
    pix = np.bincount(gt.ravel() * 5 + gt.ravel(), minlength=25).reshape(5, 5)
    assert xview2_scores(pix)["loc_f1"] == 1.0


def test_grid_summary():
    m = np.zeros((96, 96), bool); m[:48, :48] = True
    assert np.allclose(grid_summary(m, 48), [[1, 0], [0, 0]])


def test_model_forward_and_misregister():
    model = build_unet(1, pretrained=False).eval()
    p = predict(model, scene(3, 64), scene(4, 64))
    assert p.shape == (64, 64) and 0 <= p.min() <= p.max() <= 1
    x = torch.rand(2, 6, 32, 32)
    y = misregister(x, 4)
    assert torch.equal(y[:, 3:], x[:, 3:]) and not torch.equal(y[:, :3], x[:, :3])
