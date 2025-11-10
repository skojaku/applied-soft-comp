---
title: How to submit assignment
---

In this course, we will use GitHub Classroom to submit & grade assignments. Please follow the instructions below to submit your assignment.

## Option 1: A simple workflow (Full local)

See the [slides](https://docs.google.com/presentation/d/19Zvrp5kha6ohF4KvTX9W2jodKkfmsOrJfEZtO_Wg0go/edit?usp=sharing) for the detailed instructions.

1. Clone the repository from GitHub.
2. Edit the assignment.py with marimo editor. Type  `marimo edit assignment/assignment.py`
3. Submit the assignment.py via git. (You can use GitHub Desktop, or command line)
4. Check the grading on the GitHub Classroom.

## Option 2: Github Codespaces (Full cloud)

See the [slides](https://docs.google.com/presentation/d/1fVzpsgaMDFTAwNibMSIUrOOfpq6z4LJICoG8i0HvNWc/edit?usp=sharing) for the detailed instructions.

1. Go to your assignment repository on GitHub
2. Click the green "Code" button
3. Click the "Open with Codespaces" button
4. Wait for the Codespaces to be ready.
5. Type 'marimo edit assignment/assignment.py'. If you cannot find marimo, type "uv run marimo edit assignment/assignment.py" which should work.
6. You will be redirected to a webpage and prompted to enter the access token. The access token can be found on the terminal window in the Codespaces.
7. Take the access token in the url "the alphabets after "?access_token=" and enter the token in the webpage.

## Option 3: Local but with Docker Machine

See the [slides](https://docs.google.com/presentation/d/1fi5x85pW8m37eh5xIJbFrZXihM9K5BxXwmdI85h45AU/edit?usp=sharing) for the detailed instructions.

### Preparations

- Install [Docker Desktop](https://docs.docker.com/desktop/)
- Install [GitHub Desktop](https://desktop.github.com/download/)
- Install [VS Code](https://code.visualstudio.com/download)

### Steps

1. Clone the repository from GitHub.
2. Open with the VS Code, and click "Reopen in Container"
3. Open the assignment.py with marimo editor.
4. Submit the assignment.py to the repository.
