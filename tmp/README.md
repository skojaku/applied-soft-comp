# Git & Python Exercise: The Calculator Contract

## The Scenario

You've been hired as a contractor to fix and improve a broken calculator application. The previous developer left the code in a buggy state, and your job is to fix it and add new features. You'll be working alongside other contractors (your classmates), which means you'll need to coordinate your work through Git and GitHub.

## Your Mission

### Part 1: Setup & Bug Fix

1. **Fork this repository** to your own GitHub account
2. **Clone your fork** to your local machine:
   ```bash
   git clone <your-fork-url>
   cd <repo-name>
   ```

3. **Find and fix the syntax error** in `calc.py`
   - Try running the script first: `python calc.py`
   - You'll see an error. Read it carefully and fix the bug.

4. **Test your fix** to make sure the calculator works for addition, subtraction, and division.

### Part 2: Add a Feature

1. **Create a feature branch**:
   ```bash
   git checkout -b feature-multiplication
   ```

2. **Add multiplication functionality**:
   - Add a `multiply(a, b)` function to `calc.py`
   - Update the menu to include option 4 for multiplication
   - Make sure it works by testing it

3. **Commit your changes**:
   ```bash
   git add calc.py
   git commit -m "Add multiplication feature"
   ```

### Part 3: The Merge Conflict Challenge

This is where it gets interesting. Your instructor will assign you to one of two groups:

**Group A**: Change the welcome message in `welcome.py` to:
```python
message = "Welcome to the Advanced Calculator App!"
```

**Group B**: Change the welcome message in `welcome.py` to:
```python
message = "Welcome to the Python Calculator Pro!"
```

1. **Make your assigned change** to `welcome.py`
2. **Commit the change**:
   ```bash
   git add welcome.py
   git commit -m "Update welcome message"
   ```

3. **Push your branch** to GitHub:
   ```bash
   git push -u origin feature-multiplication
   ```

### Part 4: Open a Pull Request

1. Go to **your fork** on GitHub
2. Click "Compare & pull request"
3. Write a clear description of what you changed
4. Submit the PR

### Part 5: Resolve Merge Conflicts (If Assigned)

If your instructor tells you there's a conflict with another student's work:

1. **Pull the latest changes** from the main branch
2. **Merge them into your branch**:
   ```bash
   git checkout feature-multiplication
   git fetch origin
   git merge origin/main
   ```

3. **Resolve the conflict** in `welcome.py`:
   - Open the file and look for the conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`)
   - Decide which version to keep (or combine them)
   - Remove the conflict markers
   - Save the file

4. **Complete the merge**:
   ```bash
   git add welcome.py
   git commit -m "Resolve merge conflict in welcome message"
   git push
   ```

## Learning Goals

By completing this exercise, you will:

- Practice the basic Git workflow (clone, branch, commit, push)
- Learn to create and manage feature branches
- Experience merge conflicts in a safe environment
- Understand the Pull Request process
- Collaborate with others using version control

## Common Commands Cheat Sheet

```bash
# See what branch you're on
git status

# See all branches
git branch

# Switch branches
git checkout <branch-name>

# Create and switch to a new branch
git checkout -b <new-branch-name>

# See your commit history
git log --oneline

# Undo changes to a file (before staging)
git checkout -- <file-name>
```

## Need Help?

- If you're stuck on the syntax error, read the error message carefully—Python tells you exactly which line is broken
- If Git seems confusing, draw a diagram of branches on paper
- If merge conflicts are scary, remember: the conflict markers show you both versions. Your job is just to pick one (or combine them) and remove the markers

Good luck, contractor!
