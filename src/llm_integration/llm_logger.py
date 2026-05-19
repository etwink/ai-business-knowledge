"""LLM call logger — appends every request/response to logs/llm_calls.log."""

import json
import os
from datetime import datetime
from pathlib import Path


_LOG_DIR = Path("logs")
_LOG_FILE = _LOG_DIR / "llm_calls.log"

# Truncate prompt/response in the log when they exceed this character count.
# Set to 0 to log everything (large logs, but complete).
_MAX_LOG_CHARS = int(os.getenv("LLM_LOG_MAX_CHARS", "0"))


def log_call(
    method: str,
    prompt_or_messages,
    response: str,
    model: str = "",
    max_tokens: int = 0,
) -> None:
    """Append one LLM call record to the log file.

    Parameters
    ----------
    method:             "query" | "query_raw" | etc.
    prompt_or_messages: the prompt string or list of message dicts sent to the API
    response:           the text returned by the model
    model:              deployment / model name
    max_tokens:         the max_output_tokens value used
    """
    try:
        _LOG_DIR.mkdir(exist_ok=True)

        if isinstance(prompt_or_messages, list):
            prompt_repr = json.dumps(prompt_or_messages, ensure_ascii=False)
        else:
            prompt_repr = str(prompt_or_messages)

        if _MAX_LOG_CHARS > 0:
            if len(prompt_repr) > _MAX_LOG_CHARS:
                prompt_repr = prompt_repr[:_MAX_LOG_CHARS] + f"\n[... truncated at {_MAX_LOG_CHARS} chars ...]"
            if len(response) > _MAX_LOG_CHARS:
                response = response[:_MAX_LOG_CHARS] + f"\n[... truncated at {_MAX_LOG_CHARS} chars ...]"

        separator = "=" * 80
        entry = (
            f"\n{separator}\n"
            f"TIMESTAMP : {datetime.now().isoformat()}\n"
            f"METHOD    : {method}\n"
            f"MODEL     : {model}\n"
            f"MAX_TOKENS: {max_tokens}\n"
            f"--- PROMPT ({len(prompt_repr)} chars) ---\n"
            f"{prompt_repr}\n"
            f"--- RESPONSE ({len(response)} chars) ---\n"
            f"{response}\n"
        )

        with open(_LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(entry)

    except Exception:
        # Never let logging crash the application
        pass
