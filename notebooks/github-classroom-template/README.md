# Git Practice Assignment

Welcome to your Git practice assignment! This repository is where you'll submit your work for the toolkit module.

## Your Task

Complete all Git exercises from the **01-toolkit.qmd** notebook in the course materials. You'll practice:

- Making your first commits with `git add` and `git commit`
- Working with branches using `git branch` and `git checkout`
- Merging branches with `git merge`
- Resolving merge conflicts
- Pushing your work to GitHub

## Getting Started

### Step 1: Accept the Assignment and Get Your Repository

Your instructor will provide an assignment invitation link on Brightspace. When you click this link:

1. You'll be asked to accept the assignment (you may need to sign in to GitHub first)
2. GitHub Classroom will create a personal repository for you
3. After a few moments, you'll see a confirmation page with a link to your repository

Click the link to your repository. You'll be taken to your repository page on GitHub. The URL in your browser will look like:

```
https://github.com/sk-classroom/git-practice-assignment-YOUR-USERNAME
```

This is your personal repository page. Keep this page open—you'll need to copy the clone URL from here.

### Step 2: Find and Copy Your Repository Clone URL

On your repository page, look for the green **"<> Code"** button near the top right (below the repository name). Click it to open a dropdown menu.

In the dropdown:

1. You'll see three tabs: **HTTPS**, **SSH**, and **GitHub CLI**
2. Make sure **HTTPS** is selected (it should be underlined)
3. You'll see a URL that looks like: `https://github.com/sk-classroom/git-practice-assignment-YOUR-USERNAME.git`
4. Click the copy icon (📋) next to the URL to copy it to your clipboard

Now you have your repository's clone URL!

### Step 3: Clone Your Repository to Your Computer

Open your terminal and navigate to where you want to work (e.g., your Documents folder or a projects directory). Then clone your repository using the URL you just copied:

```bash
git clone https://github.com/sk-classroom/git-practice-assignment-YOUR-USERNAME.git
cd git-practice-assignment-YOUR-USERNAME
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

### Step 5: Push Your Work

After completing each major step, push your commits to GitHub:

```bash
# Push main branch
git push origin main

# Push other branches as you create them
git push origin add-median
git push origin use-numpy-mean
git push origin add-mean-docstring
```

You can push as many times as you want. Each push updates your GitHub repository.

### Step 6: Verify and Submit

Before submitting, visit your repository on GitHub (the same page from Step 1) and verify you see:

- [ ] The `analysis.py` file with both `calculate_mean()` and `calculate_median()` functions
- [ ] At least 5 commits showing your progression
- [ ] Multiple branches: `add-median`, `use-numpy-mean`, `add-mean-docstring`
- [ ] A merge commit demonstrating conflict resolution

Then submit your repository URL on Brightspace.

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
- **Push rejected:** Make sure you're inside the cloned directory (`cd git-practice-assignment-YOUR-USERNAME`)
- **Merge conflicts:** Follow the conflict resolution steps in the notebook carefully
- **General questions:** Post on the course discussion board or attend office hours

## Tips for Success

- Commit early and commit often
- Write clear, descriptive commit messages
- Push frequently so your work is backed up on GitHub
- Use `git status` often to understand what's happening
- Don't be afraid to experiment with branches—that's what they're for!

Good luck! Version control is a skill that gets easier with practice.
