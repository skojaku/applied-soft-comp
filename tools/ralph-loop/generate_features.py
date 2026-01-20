#!/usr/bin/env python3
"""
Generate expanded features from task_instructions.json.

Reads core instructions and file list, outputs a features.json
with one atomic feature per file.
"""

import json
import re
from pathlib import Path


def generate_feature_id(file_path: str, priority: int) -> str:
    """Generate a unique feature ID from file path."""
    parts = Path(file_path).parts
    category = parts[0] if parts else "misc"
    filename = Path(file_path).stem
    return f"{category}-{priority:03d}-{filename}"


def expand_pattern(pattern: str, base_dir: Path) -> list[str]:
    """
    Expand a regex pattern to matching file paths.

    Args:
        pattern: Regex pattern to match against file paths
        base_dir: Base directory to search from

    Returns:
        List of matching file paths (relative to base_dir)
    """
    regex = re.compile(pattern)
    matches = []

    for path in base_dir.rglob("*"):
        if path.is_file():
            relative = path.relative_to(base_dir)
            if regex.search(str(relative)):
                matches.append(str(relative))

    return sorted(matches)


def expand_glob(glob_pattern: str, base_dir: Path) -> list[str]:
    """
    Expand a glob pattern to matching file paths.

    Args:
        glob_pattern: Glob pattern (e.g., "m01-*/*.qmd")
        base_dir: Base directory to search from

    Returns:
        List of matching file paths (relative to base_dir)
    """
    matches = []

    for path in base_dir.glob(glob_pattern):
        if path.is_file():
            relative = path.relative_to(base_dir)
            matches.append(str(relative))

    return sorted(matches)


def generate_features(task_file: str, output_file: str) -> None:
    """Generate expanded features.json from task_instructions.json."""
    with open(task_file, "r") as f:
        tasks = json.load(f)

    meta = tasks["meta"]
    core = tasks["coreInstructions"]
    files = tasks["files"]

    # Determine directories
    script_dir = Path(__file__).parent
    project_dir = (script_dir / meta.get("projectDir", ".")).resolve()

    # For glob patterns, use script_dir as base since patterns may use relative paths
    glob_base_dir = script_dir

    # Build output structure
    output = {
        "meta": {
            "name": meta["name"],
            "description": meta["description"],
            "projectDir": meta["projectDir"],
            "structure": meta["structure"],
            "validators": meta["validators"],
            "notes": core["styleRules"],
            "writingPatterns": core.get("writingPatterns", {}),
            "learningGoalsTemplate": core.get("learningGoalsTemplate", ""),
            "figureSyntax": core.get("figureSyntax", {}),
            "codeVisibility": core.get("codeVisibility", {}),
        },
        "coreInstructions": core,
        "codeBlockRefactoring": tasks.get("codeBlockRefactoring", {}),
        "textCondensing": tasks.get("textCondensing", {}),
        "features": [],
    }

    # Generate one feature per file
    for file_entry in files:
        category = file_entry["category"]
        priority = file_entry["priority"]

        matched_paths = []

        # Check if pattern (regex) is specified
        if "pattern" in file_entry:
            pattern = file_entry["pattern"]
            matched_paths = expand_pattern(pattern, project_dir)

            if not matched_paths:
                print(f"Warning: Regex pattern '{pattern}' matched no files")
                continue

            print(f"Regex pattern '{pattern}' matched {len(matched_paths)} files")

        # Check if path contains glob patterns
        elif "path" in file_entry:
            path = file_entry["path"]

            # Detect glob patterns (*, ?, [, ])
            if any(char in path for char in ["*", "?", "[", "]"]):
                matched_paths = expand_glob(path, glob_base_dir)

                if not matched_paths:
                    print(f"Warning: Glob pattern '{path}' matched no files")
                    continue

                print(f"Glob pattern '{path}' matched {len(matched_paths)} files")
            else:
                # Explicit path - single file
                matched_paths = [path]

        # Create features for all matched files
        for idx, file_path in enumerate(matched_paths):
            feature = {
                "id": generate_feature_id(file_path, priority + idx),
                "priority": priority + idx,
                "category": category,
                "file": file_path,
                "description": f"{core['description']}: {file_path}",
                "steps": core["steps"],
                "passes": False,
            }
            output["features"].append(feature)

    # Write output
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Generated {len(output['features'])} features -> {output_file}")


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    task_file = script_dir / "task_instructions.json"
    output_file = script_dir / "features.json"

    generate_features(str(task_file), str(output_file))
