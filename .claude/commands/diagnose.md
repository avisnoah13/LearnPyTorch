Diagnose a training run that isn't converging or is producing bad predictions.

Check in order:
1. **Loss not decreasing** — learning rate too high/low? Try 10x smaller. Check `zero_grad` is called before `backward`.
2. **Loss NaN** — exploding gradients. Add `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` before `optimizer.step()`.
3. **Predictions all the same value** — model collapsed. Check normalization isn't dividing by zero (std ≈ 0 for a constant feature).
4. **Predictions in wrong range** — denormalization step missing or using wrong stats. Print raw model output and compare to normalized target range.
5. **Shape errors** — print tensor shapes at each step. Check `batch_first=True` is set on RNN layers.
