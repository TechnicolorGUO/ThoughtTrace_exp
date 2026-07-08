"""Run model inference on processed ThoughtTrace test data.

This script will load a trained base model or LoRA checkpoint, generate user
next-message predictions from fixed test-set inputs, and write prediction
records with metadata, references, and raw model outputs.
"""
