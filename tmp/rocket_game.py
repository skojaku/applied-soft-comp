#!/usr/bin/env python3
"""
Space Rocket Fuel Calculator Game
A simple terminal game where a rocket travels across the screen.
"""

import time
import sys
import os


def clear_screen():
    """Clear the terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')


def draw_rocket(position):
    """Draw the rocket at a given position."""
    rocket = "      /\\      \n     |  |     \n🔥==|  |==>>\n     |  |     \n      \\/      "
    lines = rocket.split('\n')

    screen = []
    for line in lines:
        screen.append(' ' * position + line)

    return '\n'.join(screen)


def play_game():
    """Main game loop."""
    print("=" * 50)
    print("🚀 SPACE ROCKET FUEL CALCULATOR 🚀")
    print("=" * 50)
    print("\nYour rocket is launching!")
    print("Watch it travel across space...\n")

    max_position = 30

    # Animate rocket moving across screen
    for position in range(max_position):
        clear_screen()
        print(draw_rocket(position))
        print("\n" + "=" * 50)
        print(f"Position: {position + 1}/{max_position}")
        print("=" * 50)
        time.sleep(0.2)

    print("\n🎉 SUCCESS! The rocket reached its destination!")


if __name__ == "__main__":
    try:
        play_game()
    except KeyboardInterrupt:
        print("\n\n🛑 Mission aborted by user.")
        sys.exit(0)
