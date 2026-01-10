#!/usr/bin/env python3
"""
Generate expanded features from task_instructions.json.

Reads core instructions and file list, outputs a features.json
with one atomic feature per file.
"""

import json
from pathlib import Path


def generate_feature_id(file_path: str, priority: int) -> str:
    """Generate a unique feature ID from file path."""
    parts = Path(file_path).parts
    category = parts[0] if parts else "misc"
    filename = Path(file_path).stem
    return f"{category}-{priority:03d}-{filename}"


def generate_features(task_file: str, output_file: str) -> None:
    """Generate expanded features.json from task_instructions.json."""
    with open(task_file, "r") as f:
        tasks = json.load(f)

    meta = tasks["meta"]
    core = tasks["coreInstructions"]
    files = tasks["files"]

    # Build output structure
    output = {
        "meta": {
            "name": meta["name"],
            "description": meta["description"],
            "projectDir": meta["projectDir"],
            "structure": meta["structure"],
            "validators": meta["validators"],
            "notes": core["styleRules"],
        },
        "features": [],
    }

    # Generate one feature per file
    for file_entry in files:
        file_path = file_entry["path"]
        category = file_entry["category"]
        priority = file_entry["priority"]

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
