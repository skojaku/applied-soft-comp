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


def generate_features(task_file: str, output_file: str) -> None:
    """Generate expanded features.json from task_instructions.json."""
    with open(task_file, "r") as f:
        tasks = json.load(f)

    meta = tasks["meta"]
    core = tasks["coreInstructions"]
    files = tasks["files"]

    # Determine project base directory
    script_dir = Path(__file__).parent
    project_dir = (script_dir / meta.get("projectDir", ".")).resolve()

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
        },
        "features": [],
    }

    # Generate one feature per file
    for file_entry in files:
        category = file_entry["category"]
        priority = file_entry["priority"]

        # Check if pattern is specified, otherwise use explicit path
        if "pattern" in file_entry:
            pattern = file_entry["pattern"]
            matched_paths = expand_pattern(pattern, project_dir)

            if not matched_paths:
                print(f"Warning: Pattern '{pattern}' matched no files")
                continue

            print(f"Pattern '{pattern}' matched {len(matched_paths)} files")

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
        else:
            # Use explicit path (backward compatibility)
            file_path = file_entry["path"]
            feature = {
                "id": generate_feature_id(file_path, priority),
                "priority": priority,
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
