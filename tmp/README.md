# 🚀 SPACE STATION OMEGA: The Git Mission

## Mission Briefing

Welcome, Commander. You've been assigned to a critical mission: rescue a broken space adventure game using your Git skills. The previous developer abandoned the project mid-crisis (unprofessional, we know), and now it's up to you and your team to save it.

This isn't just any exercise. This is a **playable text adventure** with real stakes, plot twists, and yes... bugs that need fixing.

## 🎮 What Makes This Fun?

- **It's an actual game** you'll want to play once it's fixed
- **Branching storylines** (see what we did there? Git branches = story branches!)
- **Merge conflicts are part of the narrative** - different timelines colliding!
- **Creative freedom** - you get to add your own plot twists
- **Instant gratification** - watch your fixes come to life immediately

## 🎯 Mission Objectives

### Phase 1: Emergency Repairs (Fix the Bug)

1. **Clone the repo** (get your space suit on):
   ```bash
   git clone <repo-url>
   cd space-station-omega
   ```

2. **Try to run the game** (it will crash):
   ```bash
   python adventure.py
   ```

3. **Find the syntax error** in `adventure.py`:
   - Read the error message carefully
   - The Python interpreter tells you exactly where the bug is
   - Fix it and try running again

4. **Play through the game** to make sure it works:
   - Choose different paths
   - Experience the story
   - Enjoy CHIP's commentary

5. **Commit your fix**:
   ```bash
   git add adventure.py
   git commit -m "Fix critical syntax error - station operational"
   ```

### Phase 2: Add Your Story Branch 🌟

Now comes the creative part! Create a new branch to add features:

1. **Create your feature branch**:
   ```bash
   git checkout -b feature-plot-twist
   ```

2. **Add your own plot twist** to the escape pod scene:
   - Find the `TODO` comment in `adventure.py` (around line 75)
   - Add one of these plot elements (or invent your own!):
     - **Alien encounter**: A friendly/hostile alien appears
     - **Rival survivor**: Another crew member who wants the last pod
     - **Robot rebellion**: A malfunctioning robot guards the pods
     - **Surprise rescue**: An unexpected savior appears
     - **Your own idea**: Get creative!

3. **Make it interactive**:
   - Add player choices (like the cafeteria scene)
   - Include CHIP's reaction
   - Use `slow_print()` for dramatic effect

4. **Test your addition**:
   ```bash
   python adventure.py
   ```

5. **Commit your new feature**:
   ```bash
   git add adventure.py
   git commit -m "Add plot twist: [describe your addition]"
   ```

### Phase 3: The Parallel Universe Collision (Merge Conflicts!) 💥

Here's where Git gets real. You and another student will edit the same part of the code, creating **parallel timelines** that must be merged.

**Your instructor will assign you to Team A, B, or C:**

#### Team A: "Sarcastic CHIP"
Change the greeting in `characters.py`:
```python
def chip_greeting():
    message = "Oh great, you're awake. I was enjoying the peace and quiet!"
    return message
```

#### Team B: "Optimistic CHIP"
Change the greeting in `characters.py`:
```python
def chip_greeting():
    message = "Good morning, Commander! Today is a BEAUTIFUL day to not die!"
    return message
```

#### Team C: "Dramatic CHIP"
Change the greeting in `characters.py`:
```python
def chip_greeting():
    message = "Commander! I've been waiting for you to wake up. We need to talk... about THE END!"
    return message
```

**After making your change:**

1. **Commit it**:
   ```bash
   git add characters.py
   git commit -m "Update CHIP's personality - Team [A/B/C]"
   ```

2. **Push your branch**:
   ```bash
   git push -u origin feature-plot-twist
   ```

3. **Open a Pull Request** on GitHub:
   - Go to the repository
   - Click "Compare & pull request"
   - Describe what you added (be creative!)
   - Submit

### Phase 4: Resolve the Timeline Collision 🔀

When your instructor merges one PR first, the others will have conflicts. If this happens to you:

1. **Fetch the latest timeline**:
   ```bash
   git fetch origin
   git merge origin/main
   ```

2. **You'll see a CONFLICT message**. Don't panic! This is normal.

3. **Open `characters.py`** and find the conflict markers:
   ```python
   <<<<<<< HEAD
   Your version
   =======
   Their version
   >>>>>>> origin/main
   ```

4. **Decide which timeline to keep** (or combine them):
   - Keep your version
   - Keep their version
   - Merge them into something new!

5. **Remove the conflict markers** (`<<<<<<<`, `=======`, `>>>>>>>`)

6. **Save, test, and commit**:
   ```bash
   python adventure.py  # Make sure it works!
   git add characters.py
   git commit -m "Resolve timeline collision - merged CHIP personalities"
   git push
   ```

## 🎓 What You're Actually Learning

Behind the fun, you're mastering real Git skills:

- **Branching**: Creating parallel development paths
- **Merging**: Combining different lines of work
- **Conflict resolution**: Handling when changes collide
- **Pull Requests**: Professional collaboration workflow
- **Commit messages**: Documenting your changes

These are the exact skills professional developers use every day.

## 🚨 Common Issues & Solutions

**"The game won't run!"**
- Read the error message - Python tells you the line number
- Look for missing colons, quotes, or parentheses

**"Git says I have conflicts!"**
- This is EXPECTED and part of the exercise
- Follow Phase 4 instructions above
- The conflict markers show both versions - pick one and remove the markers

**"I'm on the wrong branch!"**
- Check with: `git branch` (current branch has an asterisk)
- Switch with: `git checkout <branch-name>`

**"I broke everything!"**
- Undo uncommitted changes: `git checkout -- <file>`
- Go back to a previous commit: `git log` then `git checkout <commit-hash>`

## 🎬 Bonus Challenges

Finished early? Try these:

1. **Add more story branches** - create multiple endings
2. **Add inventory system** - collect items along the way
3. **Add ASCII art** - make scenes more visual
4. **Add sound effects** - use Python's `winsound` or `os.system('say')`
5. **Create a bad ending** - what if you make wrong choices?
6. **Add a scoring system** - rate the player's decisions

## 🏆 Success Criteria

You've completed the mission when:

- ✅ The game runs without errors
- ✅ You've added a creative plot twist
- ✅ You've survived a merge conflict
- ✅ Your Pull Request is merged
- ✅ You had fun (most important!)

## 💬 Git Cheat Sheet

```bash
# Where am I?
git status

# See all branches
git branch

# Create and switch to new branch
git checkout -b <branch-name>

# Switch branches
git checkout <branch-name>

# See commit history
git log --oneline --graph

# Undo uncommitted changes
git checkout -- <file>

# See what changed
git diff
```

---

**Remember**: In space, no one can hear you scream... but in Git, everyone can see your commit messages. Make them good! 🚀

Good luck, Commander. The station is counting on you.
