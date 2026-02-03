#!/usr/bin/env python3
"""
Space Rocket Fuel Calculator Game
A simple terminal game where a rocket travels across the screen.
"""

import time
import sys
import os


def play_game():
    """Main game loop."""
    print("=" * 50)
    print("🚀 SPACE ROCKET FUEL CALCULATOR 🚀")
    print("=" * 50)
    print("\nYour rocket is launching!")

    max_position = 30

    # Simple animation
    for position in range(max_position):
        print(f"Position: {position + 1}/{max_position}")
        time.sleep(0.2)

    print("\n🎉 SUCCESS! The rocket reached its destination!")


if __name__ == "__main__":
    try:
        play_game()
    except KeyboardInterrupt:
        print("\n\n🛑 Mission aborted by user.")
        sys.exit(0)
