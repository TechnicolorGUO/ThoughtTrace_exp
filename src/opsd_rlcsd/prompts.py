"""Prompt templates for ThoughtTrace RLCSD/verl OPSD."""

from src.opsd_rlcsd.opsd_format import (
    TEACHER_SYSTEM_PROMPT,
    USER_SIM_SYSTEM_PROMPT,
    build_student_user_message,
    build_teacher_user_message,
)


STUDENT_SYSTEM_MESSAGE = USER_SIM_SYSTEM_PROMPT
TEACHER_SYSTEM_MESSAGE = TEACHER_SYSTEM_PROMPT


def build_student_messages(problem: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": STUDENT_SYSTEM_MESSAGE},
        {"role": "user", "content": build_student_user_message(problem)},
    ]


def build_teacher_messages(problem: str, private_thought: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": TEACHER_SYSTEM_MESSAGE},
        {"role": "user", "content": build_teacher_user_message(problem, private_thought)},
    ]


# Kept for compatibility with the copied trainer imports.
TEACHER_PROMPT_TEMPLATE_SOLUTION_ANSWER = ""
TEACHER_PROMPT_TEMPLATE_ANSWER_ONLY = ""
