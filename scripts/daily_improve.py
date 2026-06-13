#!/usr/bin/env python3
"""Daily improvement script for code-compass.

Picks the next pending task from scripts/tasks.json, asks Claude to implement
it, runs checks, and commits + pushes if everything passes.

Usage:
    python scripts/daily_improve.py
    python scripts/daily_improve.py --dry-run       # print task, do nothing
    python scripts/daily_improve.py --task <id>     # run a specific task by id
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
TASKS_FILE = Path(__file__).parent / "tasks.json"
MAX_FILE_CHARS = 10_000  # cap per file to stay within token budget


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=check)


def git_is_clean() -> bool:
    return _run(["git", "status", "--porcelain"]).stdout.strip() == ""


def git_revert() -> None:
    subprocess.run(["git", "checkout", "--", "."], cwd=REPO_ROOT)
    subprocess.run(["git", "clean", "-fd", "--exclude=scripts/tasks.json"], cwd=REPO_ROOT)


# ---------------------------------------------------------------------------
# Task management
# ---------------------------------------------------------------------------


def load_tasks() -> list[dict]:
    return json.loads(TASKS_FILE.read_text())


def save_tasks(tasks: list[dict]) -> None:
    TASKS_FILE.write_text(json.dumps(tasks, indent=2) + "\n")


def get_task(tasks: list[dict], task_id: str | None) -> dict | None:
    for t in tasks:
        if task_id:
            if t["id"] == task_id:
                return t
        elif t["status"] == "pending":
            return t
    return None


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------


def build_context(task: dict) -> str:
    parts: list[str] = []
    for rel in task.get("context_files", []):
        path = REPO_ROOT / rel
        if not path.exists():
            continue
        content = path.read_text()[:MAX_FILE_CHARS]
        truncated = " (truncated)" if len(path.read_text()) > MAX_FILE_CHARS else ""
        parts.append(f"=== {rel}{truncated} ===\n{content}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Claude API call
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert Python developer implementing a focused, self-contained
improvement to code-compass — a RAG-powered code intelligence CLI tool.

Return ONLY a raw JSON object (no markdown fences, no prose) with this exact shape:
{
  "commit_message": "<conventional-commit subject>\\n\\n<optional body>\\n\\nCo-Authored-By: claude-bot <noreply@anthropic.com>",
  "files": [
    {"path": "relative/path/to/file.py", "content": "<full file content>"}
  ]
}

Rules:
- Include FULL content of every file you create or modify (never partial)
- Follow existing code style: ruff-formatted, no superfluous comments
- New test files go in tests/
- Commit message must follow Conventional Commits (feat/fix/test/refactor/chore)
- Make the smallest change that satisfies the task — do not refactor bystander code
"""


def call_claude(task: dict, context: str) -> dict:
    try:
        import anthropic
    except ImportError:
        print("anthropic SDK not installed. Run: pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic()

    user_prompt = f"""Task title: {task["title"]}

Task description:
{task["description"]}

Current codebase context (files you may need to read or modify):
{context if context else "(no context files specified)"}
"""

    msg = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = msg.content[0].text.strip()
    # Strip accidental markdown fences
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

    return json.loads(raw)


# ---------------------------------------------------------------------------
# Apply + verify
# ---------------------------------------------------------------------------


def apply_changes(changes: dict) -> None:
    for fc in changes["files"]:
        path = REPO_ROOT / fc["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(fc["content"])
        print(f"  wrote {fc['path']}")


def run_checks() -> bool:
    for cmd in [
        ["ruff", "check", "--fix", "."],
        ["ruff", "format", "."],
        ["python3", "-m", "pytest", "tests/", "-q", "--tb=short"],
    ]:
        result = _run(cmd, check=False)
        if result.returncode != 0:
            print(f"FAILED: {' '.join(cmd)}")
            print(result.stdout[-2000:])
            print(result.stderr[-1000:])
            return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the next code-compass daily improvement.")
    parser.add_argument("--dry-run", action="store_true", help="Print task and exit")
    parser.add_argument("--task", default=None, help="Run a specific task by ID")
    args = parser.parse_args()

    tasks = load_tasks()
    task = get_task(tasks, args.task)

    if not task:
        print("No pending tasks — all done!" if not args.task else f"Task '{args.task}' not found.")
        return

    print(f"[{task['id']}] {task['title']}")

    if args.dry_run:
        print(f"\nDescription:\n{task['description']}\n")
        print(f"Context files: {task.get('context_files', [])}")
        return

    if not git_is_clean():
        print("Working tree is not clean. Commit or stash first.")
        sys.exit(1)

    context = build_context(task)
    print("Asking Claude to implement…")

    try:
        changes = call_claude(task, context)
    except Exception as exc:
        print(f"Claude call failed: {exc}")
        sys.exit(1)

    print(f"Applying {len(changes['files'])} file(s)…")
    apply_changes(changes)

    print("Running checks…")
    if not run_checks():
        print("Checks failed — reverting.")
        git_revert()
        task["status"] = "failed"
        task["failed_at"] = datetime.now(UTC).isoformat()
        save_tasks(tasks)
        sys.exit(1)

    task["status"] = "completed"
    task["completed_at"] = datetime.now(UTC).isoformat()
    save_tasks(tasks)

    _run(["git", "add", "-A"])
    _run(["git", "commit", "-m", changes["commit_message"]])
    _run(["git", "push"])

    print(f"✓ Done: {task['title']}")


if __name__ == "__main__":
    main()
