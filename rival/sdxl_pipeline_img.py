# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch


from diffusers.utils import deprecate, logging, BaseOutput
from diffusers import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from einops import rearrange

from icecream import ic

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def adain_latent(feat, cond_feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size() # [1, 4, 1, 64, 64]
    C = size[1]
    feat_var = feat.view(C, -1).var(dim=1) + eps
    feat_std = feat_var.sqrt().view(1, C, 1, 1)
    feat_mean = feat.view(C, -1).mean(dim=1).view(1, C, 1, 1)
    
    cond_feat_var = cond_feat.view(C, -1).var(dim=1) + eps
    cond_feat_std = cond_feat_var.sqrt().view(1, C, 1, 1)
    cond_feat_mean = cond_feat.view(C, -1).mean(dim=1).view(1, C, 1, 1)
    feat = (feat - feat_mean.expand(size)) / feat_std.expand(size)
    return feat * cond_feat_std.expand(size) + cond_feat_mean.expand(size)

class RIVALStableDiffusionXLPipeline(StableDiffusionXLPipeline):
    # TBD
    pass
