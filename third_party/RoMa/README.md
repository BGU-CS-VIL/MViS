# Third-party code: RoMa

This directory contains code from **RoMa: Robust Dense Feature Matching** (Edstedt et al., CVPR 2024),
https://github.com/Parskatt/RoMa, used to estimate the ORTs (see `orts.py`).

- RoMa is released under the MIT License, Copyright (c) 2023 Johan Edstedt. See [`LICENSE`](LICENSE).
- `models/transformer/` contains DINOv2 code (Meta Platforms, Inc.), https://github.com/facebookresearch/dinov2,
  released under the Apache License 2.0. See [`models/transformer/LICENSE`](models/transformer/LICENSE).

The matcher itself is unchanged. The pretrained RoMa (outdoor) and DINOv2 ViT-L/14 weights are downloaded at
runtime from the upstream release URLs (see `models/model_zoo/__init__.py`).
