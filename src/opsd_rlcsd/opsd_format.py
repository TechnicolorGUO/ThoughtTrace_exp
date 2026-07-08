"""ThoughtTrace prompt helpers for RLCSD/verl-style OPSD."""

from __future__ import annotations


USER_SIM_SYSTEM_PROMPT = (
    "You are simulating a real user in a human-AI conversation. Given the "
    "conversation history and the assistant's latest reply, write the user's "
    "next message. Reply with only the user's next message."
)

TEACHER_SYSTEM_PROMPT = (
    "You are a thought-informed user simulator. You may use the user's private "
    "thought as privileged background, but you must output only the user's next "
    "message and must not quote or mention the private thought."
)

USER_REPLY_INSTRUCTION = "Write only the user's next message. Do not add analysis or explanation."

TEACHER_TRANSITION_PROMPT = (
    "\n\nUse the private thought above as background for intent, tone, and constraints. "
    "Now write the user's next message only.\n"
)


def _answer_instruction(thinking: bool = False) -> str:
    del thinking
    return USER_REPLY_INSTRUCTION


def build_student_user_message(problem: str) -> str:
    return f"{str(problem).strip()}\n\n{USER_REPLY_INSTRUCTION}"


def build_teacher_user_message(problem: str, private_thought: str) -> str:
    return (
        f"{str(problem).strip()}\n\n"
        "=== User Private Thought Start ===\n"
        f"{str(private_thought).strip()}\n"
        "=== User Private Thought End ==="
        f"{TEACHER_TRANSITION_PROMPT}"
    )


def normalize_privileged_text_mode(mode: str) -> str:
    normalized = str(mode or "solution").strip().lower().replace("-", "_")
    if normalized in {"solution", "solution_answer", "thought", "private_thought"}:
        return "solution"
    raise ValueError("ThoughtTrace OPSD supports only solution/private_thought privileged text.")
