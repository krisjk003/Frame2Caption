"""Builds the user-turn prompt that rewrites the factual summary into four
distinct caption styles while preserving every fact.

Pure prompt engineering — no fine-tuning involved. The factual summary
produced by the first stage is treated as immutable ground truth; this
prompt only controls tone and phrasing.
"""
from __future__ import annotations

_STYLE_GUIDE = """
- "formal": A professional, neutral caption suitable for a news brief or \
corporate report. Clear and precise, with no slang and no jokes.
- "sarcastic": A dry, witty, sarcastic caption. It should sound clever and \
knowingly amused, but every fact stated must still be true — the sarcasm \
comes from tone and framing, never from invented events.
- "humorous_tech": A funny caption written from the point of view of \
someone who loves technology, gadgets, software, or engineering. The \
humor should lean on tech-culture references or jargon-savvy, "nerdy" \
observations, while staying accurate to what happens in the video.
- "humorous_non_tech": A funny caption for a general, non-technical \
audience. Everyday, relatable humor with no jargon, accessible to anyone, \
while staying accurate to what happens in the video.
""".strip()

_OUTPUT_SCHEMA_EXAMPLE = (
    '{"formal": "...", "sarcastic": "...", "humorous_tech": "...", '
    '"humorous_non_tech": "..."}'
)


def build_rewrite_prompt(summary: str) -> str:
    """Return the instruction that rewrites a factual summary into four
    caption styles.

    Args:
        summary: The factual, hallucination-free summary produced by the
            first pipeline stage.

    Returns:
        The fully formatted prompt string, including the frozen factual
        summary and strict output-format instructions.
    """

    return f"""
Here is a verified, factual summary of a video. Treat every statement in \
it as ground truth. Do not add any event, object, person, or detail that \
is not already present in this summary:

\"\"\"{summary}\"\"\"

Rewrite this into FOUR different captions, one for each style described \
below. Every caption must:
- Stay 100% factually correct relative to the summary above.
- Be concise (roughly 1 to 3 sentences, ideally under 60 words).
- Read naturally, like something a person would actually post or say, not \
like a mechanical restatement of the summary.
- Be genuinely humorous where humor is requested, without ever inventing \
events, people, objects, or outcomes that are not in the summary.

Styles:
{_STYLE_GUIDE}

Return ONLY a single JSON object with exactly these four keys and string \
values, and nothing else — no markdown code fences, no explanation, no \
extra keys:
{_OUTPUT_SCHEMA_EXAMPLE}
""".strip()
