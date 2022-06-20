# Copyright (c) 2022 Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

# auto check if bFloat16 is supported or not

import torch
import torch.distributed as dist
import torch.cuda.nccl as nccl
from distutils.version import LooseVersion

bf16_ready = (
    torch.version.cuda
    and torch.cuda.is_bf16_supported() 
    and LooseVersion(torch.version.cuda) >= "11.0"
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
)
