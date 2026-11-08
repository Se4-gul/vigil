#!/bin/zsh
# Sequential: never training and physics at the same time. Stops at the first failure.
cd "$(dirname "$0")/.."      # repo root
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4
while pgrep -f vigil.unet_tune2 >/dev/null; do sleep 60; done
grep -q "\] done" data/processed/logs/unet_tune2.log || { echo "CHAIN STOP: tuning did not finish"; exit 1; }
echo "[$(date +%H:%M)] physics CV on fixed data"
nice -n 10 .venv/bin/python -c "from vigil.physics import cv; cv(3)" || { echo "CHAIN STOP: physics cv failed"; exit 1; }
echo "[$(date +%H:%M)] dry run incl. physics"
nice -n 10 .venv/bin/python -m vigil.test_eval --dry-run 0 || { echo "CHAIN STOP: dry run failed"; exit 1; }
echo "[$(date +%H:%M)] TEST RUN"
nice -n 10 .venv/bin/python -m vigil.test_eval || { echo "CHAIN STOP: test run failed"; exit 1; }
echo "CHAIN DONE"
