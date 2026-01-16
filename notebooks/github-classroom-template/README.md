# Git Practice Assignment

Welcome to your Git practice assignment! This repository is where you'll submit your work for the toolkit module.

## Your Task

Complete all Git exercises from the **01-toolkit.qmd** notebook in the course materials. You'll practice:

- Making your first commits with `git add` and `git commit`
- Working with branches using `git branch` and `git checkout`
- Merging branches with `git merge`
- Resolving merge conflicts
- Pushing your work to GitHub

> [!IMPORTANT]
> **Windows Users:** This course uses Unix-based command line tools. You must install Windows Subsystem for Linux (WSL) before starting.
>
> **To install WSL:**
> 1. Open PowerShell as Administrator
> 2. Run: `wsl --install`
> 3. Restart your computer
> 4. Ubuntu will open automatically after restart
> 5. Create a username and password when prompted
>
> Use the Ubuntu terminal (not PowerShell or Command Prompt) for all Git commands. Find it by searching "Ubuntu" in the Start menu.
>
> See [Microsoft's WSL installation guide](https://learn.microsoft.com/en-us/windows/wsl/install) for details.

## Getting Started

### Step 1: Accept the Assignment and Get Your Repository

Your instructor will provide an assignment invitation link on Brightspace. When you click this link:

1. You'll be asked to accept the assignment (you may need to sign in to GitHub first)
2. GitHub Classroom will create a personal repository for you
3. After a few moments, you'll see a confirmation page with a link to your repository

Click the link to your repository. You'll be taken to your repository page on GitHub. The URL in your browser will look like:

```
https://github.com/sk-classroom/01-toolkit-YOUR-USERNAME
```

This is your personal repository page. Keep this page open—you'll need to copy the clone URL from here.

### Step 2: Find and Copy Your Repository Clone URL

On your repository page, look for the green **"<> Code"** button near the top right (below the repository name). Click it to open a dropdown menu.

In the dropdown:

1. You'll see three tabs: **HTTPS**, **SSH**, and **GitHub CLI**
2. Make sure **HTTPS** is selected (it should be underlined)
3. You'll see a URL that looks like: `https://github.com/sk-classroom/01-toolkit-YOUR-USERNAME.git`
4. Click the copy icon (📋) next to the URL to copy it to your clipboard

Now you have your repository's clone URL!

### Step 3: Clone Your Repository to Your Computer

Open your terminal and navigate to where you want to work (e.g., your Documents folder or a projects directory). Then clone your repository using the URL you just copied:

```bash
git clone https://github.com/sk-classroom/01-toolkit-YOUR-USERNAME.git
cd 01-toolkit-YOUR-USERNAME
```

You can also just paste the URL if you copied it in Step 2. This command creates a local copy of the repository on your computer.

### Step 4: Complete the Exercises

Now follow the Git exercises in the **01-toolkit.qmd** notebook. You'll create an `analysis.py` file and make commits as you work through:

1. Creating your first commits
2. Fixing a bug and committing the fix
3. Creating a branch called `add-median`
4. Adding a median function and committing
5. Merging the branch
6. Creating merge conflicts with `use-numpy-mean` and `add-mean-docstring` branches
7. Resolving the conflicts

Work directly in this cloned directory. All your commits are already connected to GitHub!

### Step 5: Push Your Work to GitHub (CRITICAL!)

**⚠️ IMPORTANT:** Your instructor will review your work directly on GitHub. You MUST push all your commits and branches for your work to be graded. Work that only exists on your local computer cannot be seen or graded.

After completing each major step, push your commits to GitHub:

```bash
# Push main branch
git push origin main

# Push other branches as you create them
git push origin add-median
git push origin use-numpy-mean
git push origin add-mean-docstring
```

**Push frequently!** Every time you complete a section of the exercises, push your work. This ensures:
- Your work is backed up
- Your instructor can see your progress
- Your local repository and GitHub are in sync

### Step 6: Verify Everything is Pushed

Before you're done, verify that your local work is synchronized with GitHub. Run this command to check:

```bash
git status
```

You should see: `"Your branch is up to date with 'origin/main'"` and `"nothing to commit, working tree clean"`.

If you see unpushed commits or uncommitted changes, make sure to commit and push them!

**Final verification checklist:**

Visit your repository on GitHub (the same page from Step 1) and verify you see:

- [ ] The `analysis.py` file with both `calculate_mean()` and `calculate_median()` functions
- [ ] At least 5 commits showing your progression (click on "commits" to see the history)
- [ ] Multiple branches in the branches dropdown: `main`, `add-median`, `use-numpy-mean`, `add-mean-docstring`
- [ ] A merge commit demonstrating conflict resolution (visible in commit history)

**If anything is missing from GitHub, you need to push it!**

Your instructor will review your GitHub repository directly—there's no need to submit a URL on Brightspace.

## Checking Your Work

Use these commands to check your progress:

```bash
# See all your commits
git log --oneline --graph --all

# See all your branches
git branch -a

# Check current status
git status

# See what branches exist on GitHub
git branch -r
```

## Getting Help

If you encounter issues:

- **Can't find repository URL:** Click the green "Code" button on your repository page
- **Clone failed:** Make sure you copied the complete URL including the `.git` at the end
- **Push rejected:** Make sure you're inside the cloned directory (`cd 01-toolkit-YOUR-USERNAME`)
- **Not sure if everything is pushed:** Run `git status` and look for "Your branch is up to date with 'origin/main'"
- **Changes not appearing on GitHub:** Make sure you ran `git push origin main` and check that you're viewing the correct branch on GitHub
- **Merge conflicts:** Follow the conflict resolution steps in the notebook carefully
- **General questions:** Post on the course discussion board or attend office hours

**Remember:** Your instructor can only see what's on GitHub, not what's on your local computer. When in doubt, push!

## Tips for Success

- Commit early and commit often
- Write clear, descriptive commit messages
- Push frequently so your work is backed up on GitHub
- Use `git status` often to understand what's happening
- Don't be afraid to experiment with branches—that's what they're for!

Good luck! Version control is a skill that gets easier with practice.
