# utils/functions.py

import torch
from torch import Tensor

from .common import batchify, unbatchify, clip_grad_norms


def gather_by_index(src: Tensor, idx: Tensor, dim: int = 1, squeeze: bool = True) -> Tensor:
    """주어진 인덱스(idx)에 따라 소스 텐서(src)에서 값을 추출합니다."""
    expanded_shape = list(src.shape)
    expanded_shape[dim] = -1
    idx = idx.view(idx.shape + (1,) * (src.dim() - idx.dim())).expand(expanded_shape)
    squeeze = idx.size(dim) == 1 and squeeze
    return src.gather(dim, idx).squeeze(dim) if squeeze else src.gather(dim, idx)
