"""
Character dialogue system for Space Station Omega
This file will be the MERGE CONFLICT ZONE!
"""

# This dialogue will be edited by multiple students
CHIP_PERSONALITY = "sarcastic"  # Students will change this!

def chip_greeting():
    """
    CHIP's opening line - students will be assigned different versions
    This is where the merge conflict will happen!
    """
    # DEFAULT VERSION - Students will modify this
    message = "Hey Commander, looks like we're having a totally normal day!"
    return message

def chip_joke():
    """CHIP tells a space joke"""
    jokes = [
        "Why did the astronaut break up with their partner? They needed space!",
        "How do you organize a space party? You planet!",
        "Why did the sun go to school? To get brighter!",
    ]
    import random
    return random.choice(jokes)

def format_dialogue(character, text):
    """Format dialogue with character name"""
    return f"{character}: '{text}'"
