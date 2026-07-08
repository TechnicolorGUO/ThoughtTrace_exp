import torch


# --- User-simulation task templates (ThoughtTrace) --------------------------
# OPSD here distills "thought-informed" next-message generation into a student
# that never sees the private thought. `problem` = conversation history + the
# assistant's latest reply (student-visible). `solution` = the user's private
# thought (teacher-only privileged context).
USER_SIM_SYSTEM_PROMPT = (
    "You are simulating a real user in a human-AI conversation. Given the "
    "conversation history and the assistant's latest reply, write the user's "
    "next message. Reply with only the user's next message."
)
# Injected into the teacher prompt to expose the privileged private thought.
TEACHER_THOUGHT_TRANSITION = (
    "\n\nThe text between the markers is the user's private thought about the "
    "assistant's reply — it is background you may rely on but must not quote or "
    "mention. Using it, write the user's next message so it is consistent with "
    "that thought.\n"
)


class SelfDistillationDataCollator:
    """
    Data collator for self-distillation that creates both student and teacher inputs.

    Student: sees only the problem (with chat template)
    Teacher: sees problem + solution + transition prompt (with chat template)

    To enable batch-level operations (like original GKD), we pad prompts to the same length
    within each batch, and track the actual (unpadded) prompt lengths for loss masking.
    """

    def __init__(
        self,
        tokenizer,
        max_length=2048,
        reason_first=True,
        student_thinking=False,
        teacher_thinking=True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.reason_first = reason_first
        self.student_thinking = student_thinking
        self.teacher_thinking = teacher_thinking

        # Prompt for reasoning about the private thought before teaching
        self.reason_first_prompt = (
            "\n\nThe text above is the user's private thought about the assistant's reply. "
            "Briefly analyze what it implies about the user's intent and tone. "
            "Do NOT use <think> tags. Do NOT write the user's next message yet.\n"
        )
        # Prompt for transitioning to teaching mode after reasoning
        self.transition_prompt = TEACHER_THOUGHT_TRANSITION

        # Set padding side explicitly for consistency
        print(f"[DataCollator] Original padding_side: {self.tokenizer.padding_side}")
        self.tokenizer.padding_side = "right"
        print(f"[DataCollator] Set padding_side to: {self.tokenizer.padding_side}")
        print(f"[DataCollator] Reason first mode: {self.reason_first}")

    def __call__(self, features):

        batch_size = len(features)

        # Prepare student and teacher prompts using chat template (matching evaluation)
        student_prompts = []
        teacher_prompts = []
        teacher_reasoning_prompts = []  # NEW: for reason_first mode

        for feature in features:
            # Extract problem and solution from dataset
            # Handle different possible column names
            problem = feature["problem"]
            solution = feature["solution"]

            # Student prompt: only the problem (conversation history + assistant
            # reply). No access to the private thought.
            student_messages = [
                {"role": "system", "content": USER_SIM_SYSTEM_PROMPT},
                {"role": "user", "content": problem},
            ]

            # Apply chat template for student (matching evaluation)
            student_prompt = self.tokenizer.apply_chat_template(
                student_messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.student_thinking
            )
            student_prompts.append(student_prompt)

            if self.reason_first:
                # Reasoning prompt: ask teacher to first digest the private thought.
                reasoning_user_message = (
                    f"{problem}\n\n"
                    f"=== User Private Thought Start ===\n"
                    f"{solution}\n"
                    f"=== User Private Thought End ===\n\n"
                    f"{self.reason_first_prompt}"
                )
                reasoning_messages = [
                    {"role": "system", "content": USER_SIM_SYSTEM_PROMPT},
                    {"role": "user", "content": reasoning_user_message},
                ]
                reasoning_prompt = self.tokenizer.apply_chat_template(
                    reasoning_messages, tokenize=False, add_generation_prompt=True
                )
                teacher_reasoning_prompts.append(reasoning_prompt)

                # Teacher prompt will be constructed during training after reasoning
                # For now, create placeholder (will be replaced in training_step)
                teacher_prompts.append("")  # Placeholder
            else:
                # Teacher prompt: problem + privileged private thought.
                teacher_user_message = (
                    f"{problem}\n\n"
                    f"=== User Private Thought Start ===\n{solution}\n=== User Private Thought End ===\n"
                    f"{self.transition_prompt}"
                )
                teacher_messages = [
                    {"role": "system", "content": USER_SIM_SYSTEM_PROMPT},
                    {"role": "user", "content": teacher_user_message},
                ]

                # Apply chat template for teacher
                teacher_prompt = self.tokenizer.apply_chat_template(
                    teacher_messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.teacher_thinking
                )
                teacher_prompts.append(teacher_prompt)

        # Tokenize WITHOUT padding first to get true lengths
        student_encoded_no_pad = self.tokenizer(
            student_prompts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        student_prompt_lengths = [len(ids) for ids in student_encoded_no_pad["input_ids"]]

        # Find max lengths in this batch
        max_student_prompt_len = max(student_prompt_lengths)

        # Tokenize WITH padding to max length in batch
        student_encoded = self.tokenizer(
            student_prompts,
            padding="max_length",
            truncation=True,
            max_length=max_student_prompt_len,
            return_tensors="pt",
        )

        result = {
            "student_prompts": student_encoded["input_ids"],
            "student_prompt_attention_mask": student_encoded["attention_mask"],
            "student_prompt_length": max_student_prompt_len,  # Single value for batch!
            # Keep individual lengths for proper masking
            "student_prompt_lengths_per_example": torch.tensor(student_prompt_lengths),
        }

        if self.reason_first:
            # Tokenize reasoning prompts
            reasoning_encoded_no_pad = self.tokenizer(
                teacher_reasoning_prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            reasoning_prompt_lengths = [len(ids) for ids in reasoning_encoded_no_pad["input_ids"]]
            max_reasoning_prompt_len = max(reasoning_prompt_lengths)

            reasoning_encoded = self.tokenizer(
                teacher_reasoning_prompts,
                padding="max_length",
                truncation=True,
                max_length=max_reasoning_prompt_len,
                return_tensors="pt",
            )

            # Tokenize transition prompt (this will be appended after reasoning)
            # Don't use chat template here - just the raw text
            transition_text = f"\n{self.transition_prompt}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
            transition_encoded = self.tokenizer(
                [transition_text] * batch_size,
                padding=False,
                truncation=False,
                return_tensors="pt",
            )

            result.update(
                {
                    "teacher_reasoning_prompts": reasoning_encoded["input_ids"],
                    "teacher_reasoning_attention_mask": reasoning_encoded["attention_mask"],
                    "teacher_reasoning_prompt_length": max_reasoning_prompt_len,
                    "teacher_transition_tokens": transition_encoded["input_ids"],
                }
            )
        else:
            # Normal mode: tokenize teacher prompts
            teacher_encoded_no_pad = self.tokenizer(
                teacher_prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            teacher_prompt_lengths = [len(ids) for ids in teacher_encoded_no_pad["input_ids"]]
            max_teacher_prompt_len = max(teacher_prompt_lengths)

            teacher_encoded = self.tokenizer(
                teacher_prompts,
                padding="max_length",
                truncation=True,
                max_length=max_teacher_prompt_len,
                return_tensors="pt",
            )

            result.update(
                {
                    "teacher_prompts": teacher_encoded["input_ids"],
                    "teacher_prompt_attention_mask": teacher_encoded["attention_mask"],
                    "teacher_prompt_length": max_teacher_prompt_len,
                    "teacher_prompt_lengths_per_example": torch.tensor(teacher_prompt_lengths),
                }
            )

        return result
