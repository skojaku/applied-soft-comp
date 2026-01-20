# Toolkit Assignment

Welcome! This assignment helps you set up your development environment and learn Git fundamentals.

## What You Need to Do

Complete all exercises in the **01-toolkit.ipynb** notebook in this repository. The notebook will guide you through:

1. **Environment Setup** - Installing Python tools and packages
2. **Git Exercises** - Making commits, creating branches, merging, and resolving conflicts

> [!IMPORTANT]
> **Windows Users:** This course requires Unix-based command line tools. Install Windows Subsystem for Linux (WSL) before starting:
> 1. Open PowerShell as Administrator
> 2. Run: `wsl --install`
> 3. Restart your computer
> 4. Use the Ubuntu terminal for all commands (find it by searching "Ubuntu" in Start menu)
>
> See [Microsoft's WSL guide](https://learn.microsoft.com/en-us/windows/wsl/install) for details.

## How to Submit

Your instructor will review your work directly on GitHub. There's no need to submit a URL on Brightspace.

**To submit, you must push your work to GitHub:**

```bash
# Push your main branch
git push origin main

# Push all other branches you create
git push origin add-median
git push origin use-numpy-mean
git push origin add-mean-docstring
```

> [!IMPORTANT]
> **Push frequently as you work!** Your instructor can only see what's on GitHub. Work that exists only on your local computer cannot be graded.

**Before finishing, verify everything is pushed:**

```bash
git status
```

You should see: `"Your branch is up to date with 'origin/main'"` and `"nothing to commit, working tree clean"`

**Check on GitHub that you have:**
- [ ] `analysis.py` file with mean and median functions
- [ ] At least 5 commits
- [ ] Multiple branches: `main`, `add-median`, `use-numpy-mean`, `add-mean-docstring`
- [ ] A merge commit showing conflict resolution

If anything is missing, push it now!

## Need Help?

- **Not sure if pushed:** Run `git status` - should say "up to date with origin/main"
- **Changes not on GitHub:** Make sure you ran `git push origin main`
- **General questions:** Post on the course discussion board or attend office hours

**Remember:** Push early, push often. Your work must be on GitHub to be graded!
