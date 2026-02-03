"""
SPACE STATION OMEGA: A TEXT ADVENTURE
Your mission: Escape the failing space station before it crashes into the sun!
"""

import random
import time

def slow_print(text, delay=0.03):
    """Print text with a typing effect"""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

def show_header():
    print("\n" + "="*60)
    print("🚀  SPACE STATION OMEGA  🚀")
    print("="*60 + "\n")

# BUG ALERT: Missing colon on next line!
def intro()
    slow_print("⚠️  SYSTEM ALERT: CRITICAL FAILURE DETECTED ⚠️")
    time.sleep(0.5)
    slow_print("\nYou wake up to blaring alarms. The space station is shaking.")
    slow_print("Your AI companion, CHIP, flickers to life on your wrist display.")
    slow_print("\nCHIP: 'Commander! The station's orbit is decaying. We have')
    slow_print("       maybe 20 minutes before we burn up in the sun!'")
    slow_print("\nYou need to get to the escape pods. But there's a problem...")
    time.sleep(1)

def cafeteria_scene():
    print("\n" + "-"*60)
    slow_print("📍 LOCATION: Cafeteria")
    slow_print("\nThe cafeteria is a mess. Food packets float everywhere.")
    slow_print("You see three paths:")
    print("\n  1. 🚪 Take the main corridor (faster but unstable)")
    print("  2. 🔧 Go through the maintenance tunnels (safer but dark)")
    print("  3. 🍕 Stop for a quick snack (because why not?)")

    choice = input("\nWhat do you do? (1/2/3): ")
    return choice

def main_corridor():
    print("\n" + "-"*60)
    slow_print("You sprint down the main corridor.")
    slow_print("The lights flicker. A section of the floor collapses behind you!")
    slow_print("CHIP: 'That was close! Keep moving!'")
    time.sleep(1)
    return True

def maintenance_tunnels():
    print("\n" + "-"*60)
    slow_print("You crawl into the dark maintenance tunnel.")
    slow_print("It's cramped and creepy, but stable.")
    slow_print("You hear scratching sounds... probably just space rats.")
    slow_print("...right?")
    time.sleep(1)
    return True

def snack_break():
    print("\n" + "-"*60)
    slow_print("You grab a floating chocolate bar.")
    slow_print("CHIP: 'Seriously?! We're about to die!'")
    slow_print("You: 'I think better on a full stomach.'")
    slow_print("\n*munch munch*")
    slow_print("\nOkay, NOW you're ready to go.")
    time.sleep(1)
    return cafeteria_scene()  # Loop back

def escape_pod_room():
    print("\n" + "-"*60)
    slow_print("🎯 LOCATION: Escape Pod Bay")
    slow_print("\nYou made it! Three escape pods remain.")
    slow_print("But there's a problem (of course there is).")
    slow_print("\nCHIP: 'The launch system is offline. We need to fix it!'")
    slow_print("      'I'm detecting... wait... someone else is here?'")

    # TODO: Students will add a plot twist here!
    # Ideas: alien encounter, rival survivor, malfunctioning robot

    time.sleep(1)

def launch_sequence():
    print("\n" + "-"*60)
    slow_print("⚡ INITIATING LAUNCH SEQUENCE...")
    for i in range(3, 0, -1):
        print(f"\n      {i}...")
        time.sleep(1)
    slow_print("\n🚀 LAUNCH! 🚀")
    time.sleep(0.5)
    slow_print("\nYour pod shoots out into space just as the station")
    slow_print("begins its final descent toward the sun.")
    slow_print("\nCHIP: 'We did it! I never doubted us for a second.'")
    slow_print("You: 'You literally said we were going to die.'")
    slow_print("CHIP: 'That was Old CHIP. New CHIP is very optimistic.'")

def ending_survived():
    print("\n" + "="*60)
    slow_print("🎉 MISSION SUCCESS! 🎉")
    slow_print("\nYou survived the disaster!")
    slow_print("Your pod drifts toward a nearby rescue ship.")
    slow_print("The adventure continues...")
    print("="*60)

def main():
    show_header()
    intro()
    time.sleep(1)

    choice = cafeteria_scene()

    if choice == '1':
        if main_corridor():
            escape_pod_room()
    elif choice == '2':
        if maintenance_tunnels():
            escape_pod_room()
    elif choice == '3':
        if snack_break():
            escape_pod_room()
    else:
        slow_print("\nCHIP: 'That's not a valid choice! Just pick something!'")
        return main()

    launch_sequence()
    ending_survived()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Mission aborted. The station explodes. Game over!")
