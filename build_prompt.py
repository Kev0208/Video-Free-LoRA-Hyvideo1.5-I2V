"""
Prompt template helper used by train.py.

This is a minimal, rough template for combining a character name with motion and camera
descriptions. Feel free to add more components (style, lighting, scene, etc.) and reorder
them however you like. Just keep build_prompt(char: str) accepting the character name so
train.py can call it.
"""

import random

camera_bank = []
motion_bank = []

def build_prompt(char: str) -> str:
    motion = random.choice(motion_bank)
    camera = random.choice(camera_bank)
    return f"{char} {motion}. {camera}."
