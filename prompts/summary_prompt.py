"""Builds the user-turn prompt that requests the factual video summary.

Paired with ``prompts.system_prompt.SYSTEM_PROMPT`` for the first pipeline
stage. Pure prompt engineering — no fine-tuning involved.
"""
from __future__ import annotations


def build_summary_prompt(duration_seconds: float) -> str:
    """Return the instruction sent alongside the uploaded video file.

    Args:
        duration_seconds: The measured duration of the input video, used to
            remind the model how much footage it is responsible for
            covering.

    Returns:
        The fully formatted prompt string.
    """

    minutes = int(duration_seconds // 60)
    seconds = int(duration_seconds % 60)
    duration_label = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"

    return f"""
Watch this entire video, which is approximately {duration_label} long, \
from start to finish, before writing anything.

Produce ONE factual summary of this video that:
1. Covers the beginning, the middle, and the ending, in that order.
2. Notes every scene change and what changed at each cut.
3. Lists the important actions that happen, in strict chronological order.
4. Names the important objects, props, or on-screen elements involved.
5. Describes every person visible and what role they play in the video.
6. Notes the emotions people display, based only on visible cues.
7. Describes the setting and context, and the apparent purpose of the \
video.
8. Contains no hallucinated details — only what is clearly shown or heard.
9. Leaves out trivial or repetitive filler and focuses on what actually \
matters.

Write the summary as 4 to 8 well-formed sentences of plain prose, in \
strict chronological order, with no bullet points, no headers, and no \
commentary about these instructions.
""".strip()
