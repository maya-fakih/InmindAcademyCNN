"""
Usage (run from repo root, after training finishes):
    !uv run python scripts/log_results.py training_log.txt

Auto-detects the current git branch, appends the parsed result + full log
into docs/<branch-name>.md, commits, and pushes. No branch name hardcoded
anywhere — works identically on every branch, forever.
"""

import subprocess
import re
import sys
import os


def get_current_branch():
    result = subprocess.run(
        ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
        capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def parse_result(log_text):
    match = re.search(r'Test loss: ([\d.]+) \| Test acc: ([\d.]+)%', log_text)
    if not match:
        return None, None
    return match.groups()


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/log_results.py <path-to-log-file>")
        sys.exit(1)

    log_path = sys.argv[1]
    branch = get_current_branch()
    doc_path = f"docs/{branch}.md"

    with open(log_path, 'r') as f:
        log_text = f.read()

    test_loss, test_acc = parse_result(log_text)

    os.makedirs("docs", exist_ok=True)

    # Create the doc file with a template header if it doesn't exist yet
    if not os.path.exists(doc_path):
        with open(doc_path, 'w') as f:
            f.write(f"# {branch}\n\n## Aim\n(fill in before training)\n\n## Setup\n(fill in before training)\n")

    with open(doc_path, 'a') as f:
        f.write("\n## Result\n")
        if test_acc:
            f.write(f"Test loss: {test_loss} | Test acc: {test_acc}%\n")
        else:
            f.write("(Could not auto-parse test accuracy from log — check manually below)\n")
        f.write(f"\n## Full log\n```\n{log_text}\n```\n")

    subprocess.run(['git', 'add', doc_path], check=True)
    commit_msg = f"{branch}: test acc {test_acc}%" if test_acc else f"{branch}: training log"
    subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
    subprocess.run(['git', 'push'], check=True)

    print(f"Logged to {doc_path} and pushed. Branch: {branch}, Test acc: {test_acc}%")


if __name__ == '__main__':
    main()