"""Parse model outputs into comparable user replies.

This module will extract the final user reply from thought-formatted outputs
such as <thought>...</thought><reply>...</reply>, while also supporting
no-thought model outputs that directly contain the predicted user message.
"""

from __future__ import annotations

import re


REPLY_PATTERN = re.compile(r"<reply>\s*(.*?)\s*</reply>", re.DOTALL | re.IGNORECASE)


def extract_reply(text: str) -> str:
    """Return the user reply portion from a model output."""
    text = str(text or "").strip()
    match = REPLY_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return text
