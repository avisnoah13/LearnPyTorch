Run the full pipeline for a training script before committing:

1. Run the script end-to-end: `python documentationLearningExamples/$ARGUMENTS`
2. Confirm loss decreases over training
3. Confirm printed predictions are in physical units (denormalized)
4. `git add` the file and commit with a message describing what the model learns
