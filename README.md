# AirFit — AI Fitness at Home

This document provides a high-level overview of the backend and summarizes development progress.

---

## Available Workouts

All workouts follow this structure:
- **Warming-up**: 5 random exercises, using the lowest possible dumbbell weight where applicable.
- **Core**: The main part of the workout, created by various workout types.
- **Finale**: Fixed sequence of Squats, Push-ups, and Clean and Press.

### Workout Types

#### Classic
- The original workout mode.
- Contains 3 core blocks.
- Avoids repeating exercises from previous blocks.
- Exercises are mostly selected at random, ensuring plenty of variation.

---

## TODO

### Notes for Future Updates
- Fix loss logging after the optimizer step.
- Improve mini-batch support.
- Finish this document.

---

## Development

### v1

**06-May-2025**
- Added multiple exercise types.
- Updated attention mechanism: removed softmax (fixed issues with variable workout lengths).
- Reduced Classic workout length.
- Investigated loss discrepancies between training and evaluation:
  - Cause: PyTorch prints loss before the optimizer update.
  - Fix planned alongside improvements for mini-batch handling.

**04-May-2025**
- Started v1 development branch.

---

### v0

**04-May-2025**
- Proof of Concept working.
- Development version frozen.
