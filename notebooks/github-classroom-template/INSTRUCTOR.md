# Instructor Guide: Setting Up GitHub Classroom Assignment

This directory contains the template for the Git Practice assignment that students will complete as part of the toolkit module.

## Setup Instructions

### 1. Install GitHub CLI (if needed)

```bash
# macOS
brew install gh

# Windows
winget install --id GitHub.cli

# Or download from https://cli.github.com
```

Authenticate:

```bash
gh auth login
```

### 2. Create the Template Repository

Create a new repository in your GitHub organization using `gh`:

```bash
cd notebooks/github-classroom-template

# Initialize git repository
git init
git add .
git commit -m "Initial commit: Add Git practice assignment template"

# Create repository on GitHub (in your organization)
gh repo create sk-classroom/git-practice-assignment \
  --public \
  --source=. \
  --remote=origin \
  --push

# Mark as template repository
gh api repos/sk-classroom/git-practice-assignment \
  -X PATCH \
  -f is_template=true
```

Verify it's set as a template:

```bash
gh repo view sk-classroom/git-practice-assignment
```

You should see "Template: true" in the output.

### 3. Create GitHub Classroom Assignment

1. Go to your GitHub Classroom organization
2. Click "New Assignment"
3. Configure the assignment:
   - **Assignment title:** Git Practice Assignment
   - **Repository prefix:** git-practice-assignment
   - **Repository visibility:** Private (recommended) or Public
   - **Grant students admin access:** Yes (allows them to manage branches)
   - **Template repository:** Select the template you created above
   - **Deadline:** Set according to your course schedule

4. Copy the assignment invitation link

### 4. Share with Students

Provide students with:
1. The assignment invitation link
2. Link to the 01-toolkit.qmd notebook with all exercises
3. Instructions to complete all exercises before pushing to GitHub

## Grading Checklist

When reviewing student submissions, verify:

- [ ] Repository contains `analysis.py` file
- [ ] File includes both `calculate_mean()` and `calculate_median()` functions
- [ ] At least 5 commits present (should have more typically)
- [ ] Multiple branches exist: `add-median`, `use-numpy-mean`, `add-mean-docstring`
- [ ] Merge commit(s) visible in history
- [ ] Evidence of conflict resolution (check merge commit diff)
- [ ] Commit messages are descriptive and clear
- [ ] Code is properly formatted and functional

## Common Student Issues

**"I already completed the exercises locally"**
- Direct them to Option 2 in the README
- They can add the classroom repo as remote and push existing work

**"My push was rejected"**
- Usually happens if they didn't clone first and are trying to push to a non-empty repo
- Solution: Force push with `git push -f origin main` (after backing up their work)
- Or: Clone first, copy their work in, then commit and push

**"I don't see all my branches on GitHub"**
- They need to explicitly push each branch: `git push origin branch-name`
- Or push all branches: `git push --all origin`

**"Merge conflict help"**
- Point them back to the conflict resolution section in the notebook
- Remind them that conflicts are intentional and part of the learning

## Viewing Student Work

To clone and review a student's repository:

```bash
git clone https://github.com/sk-classroom/git-practice-assignment-STUDENT-USERNAME.git
cd git-practice-assignment-STUDENT-USERNAME
git log --oneline --graph --all  # View complete history
git branch -a  # See all branches
```

## Automated Grading (Optional)

Consider using GitHub Actions to automatically verify:
- Presence of required files
- Number of commits
- Existence of required branches
- Presence of merge commits

Example workflow could be added to `.github/workflows/check.yml` in the template.
