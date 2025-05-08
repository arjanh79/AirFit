# AirFit — AI Fitness at Home

This document provides a high-level overview of the backend and summarizes development progress.

---

## Available Workouts

All workouts follow this structure:
- **Warming-up**: 5 random exercises, using the lowest possible dumbbell weight where applicable.
- **Core**: The main part of the workout, created by various workout types.
- **Finale**: A fixed sequence of Squats, Push-ups, and Clean and Press.

### Workout Types

#### Basic
- Base workout class providing core functionality for all workout modes.
- Not intended as a standalone workout.

#### Classic
- The original and most varied workout mode.
- Contains 3 core blocks.
- Avoids repeating exercises from previous blocks.
- Exercises are mostly selected at random, ensuring plenty of variety.

#### Bosu
- A workout focused on the Bosu ball.
- Available exercises: Clean and Press, Mountain Climbers, Plank, Push-up — all performed using the Bosu.
- Each core block selects 3 unique exercises at random.

#### Combo
- A workout combining Clean and Press, Bent-Over Row, and Plank.
- Designed (with a little help from ChatGPT!) to target as many muscle groups as possible using the available exercises.
- The exercise lineup may evolve as new exercises are added.

#### Focus
- Uses PyTorch embeddings to create 3 distinct exercise clusters.
- Selects 1 exercise from each cluster.
- Repeats this block 3 times in the same order.

#### Forge
- A workout consisting of a warming-up, followed by 4 consecutive finale blocks.

#### Single
- A workout with 1 core block consisting of the same exercise, repeated 3 times. 
- The exercise is randomly selected from all available exercises.

#### Workout404
- Originally designed as a fallback in case the WorkoutFactory encountered issues.
- Includes the usual warming-up and finale, with a single core block: Push-ups, Step-ups, and Ab Crunches.

---

## TODO

### Notes for Future Updates
- Fix loss logging after the optimizer step.
  - At the moment (07-May), the difference is very small.
  - Might not be needed after implementing mini-bathes.
- Expand and finalize this document.

---

## Development

### v1

**08-May-2025**
- First implementation of mini-batches
  - Calculates adaptive batch size while ensuring the last batch isn’t too small.
  - Enforces a minimum of 8 samples in the final batch for training stability.
  - Much easier: reverse the dataset. The weighted loss ensures only minimal updates on older samples.

**07-May-2025**
- Added the 'Single' workout type.
- Database cleaned up. Removed reps column from Exercise table.
- Increased default reps to 8 to reduce rounds of fine-tuning.

**06-May-2025**
- Added multiple exercise types.
- Updated attention mechanism: removed softmax (fixed issues with variable workout lengths).
- Reduced Classic workout length.
- Investigated loss discrepancies between training and evaluation:
  - Cause: PyTorch prints loss before the optimizer update.
  - Planned fix alongside improvements for mini-batch handling.

**04-May-2025**
- Started v1 development branch.

---

### v0

**04-May-2025**
- Proof of Concept working.
- Development version frozen.
