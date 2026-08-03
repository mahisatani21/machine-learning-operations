#!/usr/bin/env bash
# Run this after a pipeline run to version the dataset it produced.
#
# What it does:
#   1. `dvc add` hashes data/breast_cancer.csv and stores the hash in
#      data/breast_cancer.csv.dvc (a tiny text pointer file, safe for git).
#      The actual CSV stays out of git (see .gitignore) — DVC tracks it in
#      its own cache instead, which you can push to remote storage later
#      with `dvc push` if you configure a remote.
#   2. `git add` + `git commit` the pointer file, so the git history itself
#      becomes the version history of your dataset: each commit says
#      exactly which data hash was in use at that point.
#
# steps/train.py reads this same .dvc file to tag each MLflow model version
# with the data hash it was trained on — that's what links a model version
# back to an exact, reproducible dataset version.
set -euo pipefail

DATA_FILE="data/breast_cancer.csv"

if [ ! -f "$DATA_FILE" ]; then
  echo "No $DATA_FILE found — run 'python run_pipeline.py' first." >&2
  exit 1
fi

dvc add "$DATA_FILE"

git add "${DATA_FILE}.dvc"
[ -f "data/.gitignore" ] && git add "data/.gitignore"
if git diff --cached --quiet; then
  echo "No changes to the dataset — nothing new to version."
else
  MSG="${1:-data: version update}"
  git commit -m "$MSG"
  echo "Committed new data version: $(git rev-parse --short HEAD)"
fi

echo "Current data hash:"
grep "md5:" "${DATA_FILE}.dvc"
