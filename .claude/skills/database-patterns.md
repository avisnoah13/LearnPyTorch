---
name: database-patterns
description: Training data storage conventions for this project
---

All training data is generated synthetically in-script — no external database. Data lives only in memory during training runs.

If persisting datasets becomes necessary, save as `.pt` files using `torch.save` / `torch.load`. No external data format (CSV, HDF5, etc.) without a clear reason.
