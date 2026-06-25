---
name: api-design
description: Conventions for nn.Module interfaces in this project
---

Each model file is self-contained: data generation, model definition, training loop, and results printing all in one script. No shared modules across files unless explicitly requested.

`nn.Module` subclasses follow the pattern in `controlFlowWeightSharingWarmup.py`:
- `__init__` defines layers only
- `forward` contains the compute graph
- A `string()` or similar method for human-readable results printing
