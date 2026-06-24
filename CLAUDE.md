# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project
Learning PyTorch progressively toward building ML models for real-time power converter control.

## Commands

```powershell
# Activate environment (required before running anything)
.venv\Scripts\activate

```

## Rules
- Reuse patterns from `documentationLearningExamples/` — those files represent mastered skills.
- No abstractions beyond what the current task requires.

## Codebase structure

`documentationLearningExamples/` contains a linear progression of exercises, each building on the last. New work should follow the patterns already established here:

The buck converter will be built in `buckConverterModel`

| File | Concept mastered |
|---|---|
| `numpyWarmup.py` | Manual backprop, gradient descent from scratch |
| `tensorWarmup.py` | PyTorch tensors, device switching |
| `tensorAutograd.py` | `requires_grad`, `loss.backward()`, `torch.no_grad()` |
| `legendrePolyAutograd.py` | Custom `torch.autograd.Function` with `ctx.save_for_backward` |
| `nnWarmup.py` | `nn.Sequential`, `nn.Linear`, `nn.MSELoss` |
| `optimWarmup.py` | `torch.optim` (RMSprop), `optimizer.step()` |
| `controlFlowWeightSharingWarmup.py` | Custom `nn.Module`, SGD with momentum, dynamic compute graphs |

### Buck Converter Simulation
Follow the implementation of the pythonBuckSim.pdf in `researchPapers`
 - Make a separate simulation file that outputs the data so it can be easily imported by PyTorch
 - Clearly explain its implementation in comments and how the data will be used to train the model

### Model
`nn.RNN(input_size=3, hidden_size=32, num_layers=1, batch_first=True)` → `nn.Linear(32, 1)`.
Use only `out[:, -1, :]` — this is a many-to-one RNN, not many-to-many.

### Training
`nn.MSELoss` + `torch.optim.Adam(lr=1e-3)`.
Use `torch.no_grad()` for test evaluation (same pattern as `tensorAutograd.py`).
