"""LLM providers that call external CLI tools via subprocess.

ClaudeCodeProvider: calls `claude -p "..." --output-format json --no-session-persistence`
  Works with Claude Code (claude.ai/claude-code) authenticated via OAuth.
  No API key required — uses the user's existing session.

  NOTE: do NOT use --bare. That flag disables OAuth and requires ANTHROPIC_API_KEY.

SubprocessProvider: generic template. User sets llm.cmd_template in config.
  Example: "claude -p {prompt} --output-format json"
           "ollama run llama3 {prompt}"
"""

import json
import logging
import os
import shlex
import subprocess

from codecompass.providers.base import LLMProvider

logger = logging.getLogger(__name__)

CLAUDE_CMD = ["claude", "--output-format", "json", "--no-session-persistence"]


def _messages_to_prompt(messages: list[dict], system: str | None) -> str:
    """Convert messages list to a single string prompt."""
    parts = []
    if system:
        parts.append(f"[System]: {system}")
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            parts.append(content)
        elif role == "assistant":
            parts.append(f"[Assistant]: {content}")
    return "\n\n".join(parts)


class ClaudeCodeProvider(LLMProvider):
    """Uses the `claude` CLI (Claude Code) via subprocess. No API key required."""

    def __init__(self, model: str = "", timeout: int = 120) -> None:
        self._model = model
        self._timeout = timeout

    def complete(
        self, messages: list[dict], *, max_tokens: int = 2048, system: str | None = None
    ) -> str:
        prompt = _messages_to_prompt(messages, system)
        cmd = list(CLAUDE_CMD) + ["-p", prompt]
        if self._model:
            cmd += ["--model", self._model]

        # Strip ANTHROPIC_API_KEY so the claude CLI uses OAuth, not a stale/invalid key.
        # If ANTHROPIC_API_KEY is present in the environment the CLI prefers it over OAuth,
        # which causes a 401 when the key is expired or belongs to a different account.
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                env=env,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                "claude CLI not found. Install Claude Code: https://claude.ai/claude-code"
            ) from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"claude CLI timed out after {self._timeout}s") from e

        if result.returncode != 0:
            # stdout may contain a JSON error payload with more detail than stderr
            detail = result.stderr.strip() or result.stdout.strip()
            try:
                payload = json.loads(result.stdout)
                if payload.get("result"):
                    detail = payload["result"]
            except (json.JSONDecodeError, AttributeError):
                pass
            raise RuntimeError(f"claude CLI error (exit {result.returncode}): {detail[:500]}")

        # Claude CLI --output-format json returns {"type":"result","result":"...","is_error":false,...}
        try:
            data = json.loads(result.stdout)
            if data.get("is_error"):
                raise RuntimeError(
                    f"claude returned an error: {data.get('result', result.stdout)[:500]}"
                )
            return data.get("result") or data.get("content") or str(data)
        except json.JSONDecodeError:
            return result.stdout.strip()


class SubprocessProvider(LLMProvider):
    """Generic subprocess provider. Configure with llm.cmd_template in config.

    The template may contain {prompt} which is replaced with the actual prompt.
    Output is parsed as JSON if possible ({"result": "..."} or {"content": "..."}),
    otherwise treated as plain text.
    """

    def __init__(self, cmd_template: str, timeout: int = 120) -> None:
        self._cmd_template = cmd_template
        self._timeout = timeout

    def complete(
        self, messages: list[dict], *, max_tokens: int = 2048, system: str | None = None
    ) -> str:
        prompt = _messages_to_prompt(messages, system)
        cmd_str = self._cmd_template.replace("{prompt}", shlex.quote(prompt))
        cmd = shlex.split(cmd_str)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self._timeout)
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Command not found: {cmd[0]}. Check llm.cmd_template in config."
            ) from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"Subprocess timed out after {self._timeout}s") from e

        if result.returncode != 0:
            raise RuntimeError(
                f"Subprocess error (exit {result.returncode}): {result.stderr[:500]}"
            )

        output = result.stdout.strip()
        try:
            data = json.loads(output)
            return data.get("result") or data.get("content") or data.get("text") or str(data)
        except json.JSONDecodeError:
            return output
