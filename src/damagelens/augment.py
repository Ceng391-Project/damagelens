import torch
import torch.nn.functional as F


def flip_rotate(x: torch.Tensor, y: torch.Tensor):
    if torch.rand(1) < 0.5: x, y = x.flip(-1), y.flip(-1)
    if torch.rand(1) < 0.5: x, y = x.flip(-2), y.flip(-2)
    k = int(torch.randint(4, (1,)))
    return torch.rot90(x, k, (-2, -1)).contiguous(), torch.rot90(y, k, (-2, -1)).contiguous()


def misregister(x: torch.Tensor, max_px: int) -> torch.Tensor:
    # only the pre channels move: labels are drawn on the post image
    if max_px <= 0:
        return x
    b, _, h, w = x.shape
    d = (torch.rand(b, 2, device=x.device) * 2 - 1) * max_px
    theta = torch.zeros(b, 2, 3, device=x.device); theta[:, 0, 0] = theta[:, 1, 1] = 1
    theta[:, 0, 2], theta[:, 1, 2] = 2 * d[:, 0] / w, 2 * d[:, 1] / h
    grid = F.affine_grid(theta, (b, 3, h, w), align_corners=False)
    pre = F.grid_sample(x[:, :3], grid, mode="bilinear", padding_mode="border", align_corners=False)
    return torch.cat([pre, x[:, 3:]], 1)
