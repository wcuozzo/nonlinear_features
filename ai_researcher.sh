#!/bin/bash

MAX_ITERATIONS=20
TIMEOUT_MINUTES=20
RUN_DATE=$(date +"%Y-%m-%d_%H%M")
RESULTS_DIR="./results_${RUN_DATE}"
EXPERIMENT_LOG="$RESULTS_DIR/experiment_log.md"

mkdir -p "$RESULTS_DIR"

# Initialize the experiment log if it doesn't exist
if [ ! -f "$EXPERIMENT_LOG" ]; then
  cat > "$EXPERIMENT_LOG" << 'EOF'
# Overnight Experiment Log

## Project Context
Researching nonlinear features in bottleneck autoencoders.
Investigating when/why networks learn nonlinear vs linear encoding strategies.
Key parameters: input dimensionality (n), hidden dimensions (m), network depth (l).
Goal: map phase transitions between linear and nonlinear encoding regimes.

## Experiments
EOF
fi

echo "Starting overnight loop at $(date)" | tee -a "$RESULTS_DIR/loop_log.txt"

for i in $(seq 1 $MAX_ITERATIONS); do
  echo ""
  echo "=== Iteration $i starting at $(date) ===" | tee -a "$RESULTS_DIR/loop_log.txt"

  gtimeout ${TIMEOUT_MINUTES}m claude -p "
You are running iteration $i of an autonomous overnight research loop.

PROJECT: Nonlinear features in bottleneck autoencoders.
CODEBASE: Look around the project directory. You have a BottleneckAutoencoder class,
sparse feature generation, and metrics for encoding linearity
(testing encode(ax+by) ≈ a*encode(x)+b*encode(y)), effective dimensionality,
and encoding sparsity.

PRIOR WORK: Read $EXPERIMENT_LOG carefully. Do NOT repeat experiments that have
already been run. Build on what has been learned so far.

YOUR TASK:
1. Based on prior results (or starting fresh if this is iteration 1), identify the
   single most informative experiment to run next. Think about what region of the
   (n, m, l) parameter space is least explored, or what hypothesis from prior
   experiments most needs testing.
2. Implement and run the experiment.
3. Save any plots/artifacts to $RESULTS_DIR/experiment_${i}/
4. Append results to $EXPERIMENT_LOG in this format:

   ### Experiment $i: [short title]
   - **Parameters**: n=?, m=?, l=?, plus any other relevant settings
   - **Hypothesis**: what you expected and why
   - **Result**: what actually happened (include key metrics)
   - **Implication**: what this tells us about the phase diagram
   - **Suggested next**: what would be most valuable to try next

TIME BUDGET: Keep this iteration under 15 minutes total. Use small models,
limited epochs (50-200 max), and small datasets to get directional signal fast.
If a training run exceeds 5 minutes, stop it early, log partial results, and
move on. Speed and coverage matter more than precision right now — we are
mapping the landscape, not publishing final numbers.

Do NOT ask for clarification. Just pick the best next experiment and run it.
" \
    --allowedTools "Bash(*),Read,Write,Edit" \
    >> "$RESULTS_DIR/loop_log.txt" 2>&1

  exit_code=$?

  if [ $exit_code -eq 124 ]; then
    echo "### Experiment $i: TIMED OUT after ${TIMEOUT_MINUTES}m" >> "$EXPERIMENT_LOG"
    echo "Iteration $i timed out at $(date)" | tee -a "$RESULTS_DIR/loop_log.txt"
  else
    echo "Iteration $i completed at $(date) (exit code: $exit_code)" | tee -a "$RESULTS_DIR/loop_log.txt"
  fi

  # Brief pause between iterations
  sleep 5
done

echo ""
echo "=== All iterations complete. Generating final summary at $(date) ===" | tee -a "$RESULTS_DIR/loop_log.txt"

# Final summary pass
gtimeout 10m claude -p "
Read $EXPERIMENT_LOG which contains results from an overnight autonomous research loop
on nonlinear features in bottleneck autoencoders.

Write $RESULTS_DIR/FINAL_SUMMARY.md containing:
1. A ranked list of the most interesting/important findings
2. An ASCII or text description of the emerging phase diagram
3. Which (n, m, l) regions clearly favor linear vs nonlinear encoding
4. The strongest evidence for or against the core hypothesis
5. Recommended next experiments for a human to review and prioritize

Be concise and direct. Focus on what we learned, not what we did.
" \
  --allowedTools "Read,Write" \
  >> "$RESULTS_DIR/loop_log.txt" 2>&1

echo "Done at $(date). Check $RESULTS_DIR/FINAL_SUMMARY.md" | tee -a "$RESULTS_DIR/loop_log.txt"
