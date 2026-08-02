# How to run a training branch on Kaggle

## One-time setup (only needed once ever, per Kaggle account)
1. Create a Kaggle account, verify phone number (required for GPU + internet access).
2. New Notebook → right sidebar → Settings → Accelerator: GPU (T4 x2 or P100) → Internet: ON.
3. Notebook menu → Add-ons → Secrets → add secret named `GH_TOKEN` with your GitHub
   fine-grained personal access token (Contents: Read and write, scoped to this repo only).

## Every session — one cell, replace BRANCH with the branch you want to run

```python
from kaggle_secrets import UserSecretsClient
import os

BRANCH = "05-flexresnet-w4d34-100epochs"  # <- change this per run

user_secrets = UserSecretsClient()
os.environ['GH_TOKEN'] = user_secrets.get_secret("GH_TOKEN")

!git clone https://$GH_TOKEN@github.com/maya-fakih/InmindAcademyCNN.git
%cd /kaggle/working/InmindAcademyCNN
!git checkout {BRANCH}

!git config user.email "kaggle@runner.local"
!git config user.name "Kaggle Runner"

!pip install -q uv
!uv sync

!uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

!uv run python train.py 2>&1 | tee training_log.txt
!uv run python scripts/generate_report.py
```

That's it — clone, checkout, git identity, deps, GPU check, train, generate report
(plots + concise doc), commit, push. Fully automated, no manual steps after this.

## Known gotchas (already solved, but good to remember)
- **Matplotlib backend error inside Kaggle**: if you ever see
  `ValueError: Key backend: 'module://matplotlib_inline...'`, it means
  `scripts/generate_report.py` is missing the `os.environ.pop('MPLBACKEND', None)`
  fix near the top — should already be in the committed version, just noting why
  it exists if it ever resurfaces on a fresh script.
- **If a Colab-trained checkpoint needs converting** (old raw state_dict format,
  from before resume support existed), run once:
  ```python
  !uv run python scripts/convert_old_checkpoint.py <input.pth> <output-best.pth> <epoch_num> <best_val_acc>
  ```
  Then set `resume_from` in `config.yaml` to the new `<output-best.pth>` path.
- **Never resume a checkpoint across a different platform/session without a
  seeded val split** — this caused real val-accuracy contamination on branch 04
  (jumped to 96% because Kaggle drew a different random split than Colab had).
  `get_loaders()` now uses `torch.Generator().manual_seed(42)` — keep it that way.
- **Kaggle has no Google Drive** — `train_dir`/`test_dir` should point at a local
  path (e.g. `data/cifar10/train`), not `/content/drive/...`. CIFAR-10 will
  redownload fresh each session (~170MB, a few minutes) — accepted cost, not
  worth fighting Kaggle's read-only `/kaggle/input/` dataset format matching.
- **Git identity must be set** before any commit in a fresh session
  (`git config user.email` / `user.name`) or commits fail with
  "Please tell me who you are."

## Branch naming convention
`NN-short-description` — e.g. `05-flexresnet-w4d34-100epochs` means branch 05,
FlexResNet architecture, width_factor=4, 34 total layers, 100 epochs planned.
Rename with `git branch -m old new`, then `git push -u origin new` +
`git push origin --delete old` if already pushed under the old name.