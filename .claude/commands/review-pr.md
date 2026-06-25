Review the current branch diff for correctness issues specific to this PyTorch learning project.

Check:
1. Gradient flow: `zero_grad` before backward, `no_grad` during eval
2. Tensor shapes: trace shapes through the full forward pass
3. Physics: simulation equations match the CCM model in CLAUDE.md
4. No new dependencies outside PyTorch standard library
5. Normalization stats computed on train split only

Report findings as a short bulleted list. Flag blockers separately from suggestions.
