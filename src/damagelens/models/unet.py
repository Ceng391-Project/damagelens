import json
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch

MEAN = torch.tensor([0.485, 0.456, 0.406] * 2).view(1, 6, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225] * 2).view(1, 6, 1, 1)


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def build_unet(classes: int = 1, encoder: str = "resnet18", pretrained: bool = True) -> torch.nn.Module:
    return smp.Unet(encoder, encoder_weights="imagenet" if pretrained else None, in_channels=6, classes=classes)


def normalize(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] == 6:
        x = x.permute(0, 3, 1, 2)
    return (x.float() / 255 - MEAN.to(x.device)) / STD.to(x.device)


@torch.no_grad()
def predict(model: torch.nn.Module, pre: np.ndarray, post: np.ndarray, batch: int = 8, tta: bool = False) -> np.ndarray:
    single = pre.ndim == 3
    pre, post = (pre[None], post[None]) if single else (pre, post)
    dev = next(model.parameters()).device; model.eval(); out = []
    for i in range(0, len(pre), batch):
        x = normalize(torch.from_numpy(np.ascontiguousarray(np.concatenate([pre[i:i + batch], post[i:i + batch]], -1))).to(dev))
        logits = model(x)
        if tta:
            logits = (logits + model(x.flip(-1)).flip(-1)) / 2
        p = torch.sigmoid(logits)[:, 0] if logits.shape[1] == 1 else logits.softmax(1)
        out.append(p.cpu().numpy())
    out = np.concatenate(out)
    return out[0] if single else out


def save_model(model: torch.nn.Module, path: Path, **meta) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    json.dump(meta, open(path.with_suffix(".json"), "w"), indent=2)


def load_model(path: Path, dev: str | None = None):
    path = Path(path); meta_path = path.with_suffix(".json")
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    model = build_unet(meta.get("classes", 1), meta.get("encoder", "resnet18"), pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(dev or device()).eval(), meta
