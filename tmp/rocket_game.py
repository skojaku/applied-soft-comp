#!/usr/bin/env python3
"""
Space Rocket Fuel Calculator Game
A simple terminal game where a rocket travels across the screen.
"""

import time
import sys
import os


def calculate_fuel_cost(distance, fuel_rate):
    """Calculate the fuel needed for a given distance."""
    return distance * fuel_rate


def calculate_remaining_fuel(current_fuel, fuel_used):
    """Calculate remaining fuel after consumption."""
    return current_fuel - fuel_used


def clear_screen():
    """Clear the terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')


def draw_rocket(position):
    """Draw the rocket at a given position."""
    rocket = "   ___\n  |🚀|>>>\n  |___|_\n  O   O"
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

    # Game parameters
    starting_fuel = 100
    fuel_rate = 2  # fuel units per position
    max_position = 30

    current_fuel = starting_fuel

    # Animate rocket moving across screen
    for position in range(max_position):
        clear_screen()

        # Draw rocket
        print(draw_rocket(position))
        print("\n" + "=" * 50)

        # Calculate fuel consumption
        fuel_used = calculate_fuel_cost(1, fuel_rate)
        current_fuel = calculate_remaining_fuel(current_fuel, fuel_used)

        # Display fuel info
        print(f"Position: {position + 1}/{max_position}")
        print(f"Fuel used this step: {fuel_used} units")
        print(f"Remaining fuel: {current_fuel} units")
        print("=" * 50)

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
