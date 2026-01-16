# Git Practice Assignment

Welcome to your Git practice assignment! This repository is where you'll submit your work for the toolkit module.

## Your Task

Complete all exercises from the **01-toolkit.qmd** notebook in the course materials. This includes:

1. **Environment Setup**
   - Create a virtual environment using `uv`
   - Install all required packages
   - Verify installation by running the import test

2. **Git Fundamentals**
   - Create a local `git-practice` repository
   - Make your first commits
   - Work with branches
   - Practice merging
   - Resolve merge conflicts

3. **Push to GitHub**
   - Connect your local repository to this GitHub Classroom repository
   - Push all your work, including branches and merge commits

## Getting Started

You have two options for working with this repository:

### Option 1: Start Fresh (Recommended if you haven't started yet)

Clone this repository and work directly in it:

```bash
git clone <YOUR-REPO-URL>
cd git-practice-assignment-<your-username>
git init  # Initialize as a git repository if needed
```

Then follow all the exercises from the notebook, creating your `analysis.py` file and making commits as instructed.

### Option 2: Connect Your Existing Work (If you already completed exercises locally)

If you've already completed the exercises in a local `git-practice` directory, you can connect that repository to this GitHub Classroom repository:

**Step 1:** Navigate to your local `git-practice` directory:

```bash
cd git-practice
```

**Step 2:** Add this GitHub Classroom repository as your remote:

```bash
git remote add origin <YOUR-REPO-URL>
```

If you get an error that `origin` already exists, remove the old remote first:

```bash
git remote remove origin
git remote add origin <YOUR-REPO-URL>
```

**Step 3:** Push your main branch:

```bash
git branch -M main
git push -u origin main
```

**Step 4:** Push all your other branches:

```bash
git push origin add-median
git push origin use-numpy-mean
git push origin add-mean-docstring
```

**Step 5:** Verify everything is uploaded:

Visit this repository on GitHub and check that you see:
- Your `analysis.py` file
- All your commits
- All your branches

## What Should Be in Your Repository

Your instructor will verify that your repository contains:

- [ ] An `analysis.py` file with both `calculate_mean()` and `calculate_median()` functions
- [ ] At least 5 commits showing your progression
- [ ] Multiple branches: `add-median`, `use-numpy-mean`, `add-mean-docstring`
- [ ] A merge commit demonstrating conflict resolution
- [ ] All branches pushed to GitHub

## Submission

Once you've pushed all your work to this repository:

1. Verify everything is visible on GitHub
2. Copy the URL of this repository
3. Submit the URL on Brightspace

## Getting Help

If you encounter issues:

- **Git errors:** Review the Git section in the 01-toolkit notebook
- **Push rejected:** Make sure you're in the correct directory and have added the right remote URL
- **Merge conflicts:** Follow the conflict resolution steps in the notebook
- **General questions:** Post on the course discussion board or attend office hours

## Tips for Success

- Commit early and commit often
- Write clear, descriptive commit messages
- Don't be afraid to experiment with branches
- Use `git status` frequently to understand what's happening
- Use `git log --oneline --graph --all` to visualize your branch structure

Good luck! Version control is a skill that gets easier with practice.
