# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Import from the modified build_sam to include vit_t support
from .build_sam_modified import (
    build_sam,
    build_sam_vit_h,
    build_sam_vit_l,
    build_sam_vit_b,
    build_sam_vit_t,  # Include vit_t
    sam_model_registry,
)
from .predictor import SamPredictor
from .automatic_mask_generator import SamAutomaticMaskGenerator