#!/bin/bash
set -e

MESSAGE="${1:-data: update dataset}"

# Find the Git repository root
GIT_ROOT=$(git rev-parse --show-toplevel)

# Move to the repository root
cd "$GIT_ROOT"

# Path to the project inside the repository
PROJECT_DIR="zenml_swarm_project"

# Ensure DVC is initialized
if [ ! -d ".dvc" ] && [ ! -d "$PROJECT_DIR/.dvc" ]; then
    echo "ERROR: DVC is not initialized."
    exit 1
fi

# Run DVC from the project directory
cd "$PROJECT_DIR"

dvc add data/breast_cancer.csv

# Return to the Git root
cd "$GIT_ROOT"

git add \
    "$PROJECT_DIR/data/breast_cancer.csv.dvc" \
    "$PROJECT_DIR/.gitignore"

git commit -m "$MESSAGE"

echo "✓ Dataset versioned successfully."
