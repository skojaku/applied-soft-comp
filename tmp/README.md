# Git Practice: The Broken Rocket Game 🚀

## The Situation

Welcome, Space Engineer! You've just joined our team, and we have a problem. Our rocket fuel calculator game was working perfectly, but after someone added a new feature, the fuel calculations went completely wrong. The rocket now has unlimited fuel, which breaks the entire challenge!

Your mission: Use Git to investigate what changed and fix the broken math functions.

## Getting Started

1. **Run the game to see the problem:**
   ```bash
   python3 rocket_game.py
   ```
   Notice anything weird about the fuel calculations?

2. **Your task:** The game has two simple math functions that are broken:
   - `calculate_fuel_cost()` - Should multiply distance by fuel rate
   - `calculate_remaining_fuel()` - Should subtract fuel used from current fuel

## Git Detective Work

Here are the Git commands that will help you investigate:

### Step 1: See the commit history
```bash
git log --oneline
```
This shows all commits. Look for what changed recently.

### Step 2: See what changed in a specific commit
```bash
git show <commit-hash>
```
Replace `<commit-hash>` with the hash from the log (the short string like `a1b2c3d`).

### Step 3: Compare current version to an older version
```bash
git diff <old-commit-hash> <new-commit-hash> rocket_game.py
```

### Step 4: See line-by-line history of the file
```bash
git blame rocket_game.py
```
This shows who changed each line and when.

## Your Mission

1. Use the Git commands above to find which commit broke the math functions
2. Identify what the code looked like before it broke
3. Fix the two functions in `rocket_game.py`:
   - Line ~14: `calculate_fuel_cost()`
   - Line ~19: `calculate_remaining_fuel()`
4. Test that the game works correctly
5. Commit your fix!

## Expected Behavior

When working correctly:
- The rocket should start with 100 fuel units
- Each move should cost 2 fuel units (1 position × 2 fuel_rate)
- The rocket should run out of fuel before reaching the destination
- Final fuel should be negative, showing mission failed

Good luck, Space Engineer! 🛠️
