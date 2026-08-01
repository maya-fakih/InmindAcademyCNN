#!/bin/bash
# The whole shebang, one command, works identically for every branch forever.
#
# Usage inside a Colab cell:
#   import os
#   os.environ['GH_TOKEN'] = userdata.get('GH_TOKEN')
#   !bash <(curl -s https://raw.githubusercontent.com/maya-fakih/InmindAcademyCNN/main/scripts/run_all.sh) <branch-name>
#
# Or, if the repo is already cloned in this session:
#   !bash scripts/run_all.sh <branch-name>

set -e

BRANCH="$1"
GITHUB_USER="maya-fakih"
REPO_NAME="InmindAcademyCNN"

if [ -z "$BRANCH" ]; then
  echo "Usage: bash run_all.sh <branch-name>"
  exit 1
fi

if [ -z "$GH_TOKEN" ]; then
  echo "GH_TOKEN not set. Run this in the notebook first:"
  echo "  from google.colab import userdata"
  echo "  import os; os.environ['GH_TOKEN'] = userdata.get('GH_TOKEN')"
  exit 1
fi

# --- Clone or update ---
if [ -d "$REPO_NAME" ]; then
  echo ">>> Repo exists, pulling latest on $BRANCH"
  cd "$REPO_NAME"
  git fetch origin
  git checkout "$BRANCH"
  git pull origin "$BRANCH"
else
  echo ">>> Cloning $BRANCH fresh"
  git clone -b "$BRANCH" "https://${GH_TOKEN}@github.com/${GITHUB_USER}/${REPO_NAME}.git"
  cd "$REPO_NAME"
fi

# --- Set git identity for commits made from Colab (only needed once per session) ---
git config user.email "colab@runner.local"
git config user.name "Colab Runner (Maya)"

# --- Install deps ---
echo ">>> Installing uv + syncing deps"
pip install -q uv
uv sync

# --- Confirm GPU ---
echo ">>> Checking GPU"
uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# --- Train, capturing full log ---
echo ">>> Training on branch: $BRANCH"
uv run python train.py 2>&1 | tee training_log.txt

# --- Log results back into docs/<branch>.md, commit, push ---
echo ">>> Logging results and pushing"
uv run python scripts/log_results.py training_log.txt

echo ">>> Done. Branch $BRANCH results are committed and pushed."