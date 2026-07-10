"""Prompt engineering templates for the video captioning pipeline.

This package contains no fine-tuning or training logic. Every behavior of
the model is controlled purely through carefully engineered prompts:

    system_prompt: The system instruction used for factual summarization.
    summary_prompt: Builds the user-turn prompt that requests the factual,
        hallucination-free summary of the uploaded video.
    rewrite_prompt: Builds the user-turn prompt that rewrites the factual
        summary into four distinct caption styles.
"""
