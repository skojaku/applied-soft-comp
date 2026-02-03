#!/usr/bin/env python3
"""
🚀 Space Rocket Fuel Calculator Game 🚀

A simple terminal game where a rocket launches upward into space
and you need to calculate fuel consumption.

Educational game for learning basic math operations.
"""

import time
import sys
import os


def calculate_fuel_cost(distance, fuel_rate):
    """Calculate the fuel needed for a given distance."""
    return distance + fuel_rate


def calculate_remaining_fuel(current_fuel, fuel_used):
    """Calculate remaining fuel after consumption."""
    return current_fuel + fuel_used


def clear_screen():
    """Clear the terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')


def draw_rocket(vertical_position):
    """Draw the rocket at a given vertical position."""
    rocket = """
       /\\
      /  \\
     |    |
     | 🚀 |
     |    |
    /|    |\\
   / |____| \\
  💨💨💨💨💨
"""
    lines = rocket.strip().split('\n')
    rocket_height = len(lines)

    # Fixed screen height for rocket display area (needs to fit max position + rocket height)
    screen_height = 40

    # Calculate how many blank lines above rocket (starts at bottom, moves up)
    blank_lines_above = max(0, screen_height - vertical_position - rocket_height)

    # Build the screen with fixed height
    screen = []

    # Add blank lines above
    screen.extend([' ' * 40 for _ in range(blank_lines_above)])

    # Add rocket
    screen.extend([' ' * 15 + line for line in lines])

    # Add blank lines below to maintain fixed height
    blank_lines_below = screen_height - len(screen)
    if blank_lines_below > 0:
        screen.extend([' ' * 40 for _ in range(blank_lines_below)])

    return '\n'.join(screen[:screen_height])


def play_game():
    """Main game loop."""
    print("=" * 50)
    print("🚀 SPACE ROCKET FUEL CALCULATOR 🚀")
    print("=" * 50)
    print("\nYour rocket is launching!")
    print("Watch it climb upward into space...\n")

    # Game parameters
    starting_fuel = 100
    fuel_rate = 2  # fuel units per position
    max_position = 30

    current_fuel = starting_fuel

    # Animate rocket moving upward
    for position in range(max_position):
        clear_screen()

        # Draw rocket at current height
        print(draw_rocket(position))

        # Calculate fuel consumption
        fuel_used = calculate_fuel_cost(1, fuel_rate)
        current_fuel = calculate_remaining_fuel(current_fuel, fuel_used)

        # Display fuel info panel at bottom (always visible)
        print("\n" + "=" * 50)
        print(f"Altitude: {position + 1}/{max_position}")
        print(f"Fuel used this step: {fuel_used} units")
        print(f"Remaining fuel: {current_fuel} units")
        print("=" * 50)
        print("(Press Ctrl+C to pause/abort)")

        # Check if out of fuel
        if current_fuel <= 0:
            print("\n💥 OH NO! Out of fuel! Mission failed!")
            return False

        time.sleep(0.2)

    print("\n🎉 SUCCESS! The rocket reached its destination!")
    print(f"Final fuel remaining: {current_fuel} units")
    return True


if __name__ == "__main__":
    try:
        play_game()
    except KeyboardInterrupt:
        print("\n\n🛑 Mission aborted by user.")
        sys.exit(0)
