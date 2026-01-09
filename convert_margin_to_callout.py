#!/usr/bin/env python3
"""
Convert column-margin blocks to callout blocks in Quarto documents.

This script finds all :::: {.column-margin} blocks and converts them to
:::: {.callout-note} blocks for better integration with the lecture notes.
"""

import re
from pathlib import Path


def convert_margin_to_callout(content: str) -> str:
    """
    Convert column-margin divs to callout-note divs.

    Handles both ::: and :::: style divs.
    """
    # Replace :::: {.column-margin} with :::: {.callout-note}
    content = re.sub(
        r'^(::::?)\s*\{\.column-margin\}',
        r'\1 {.callout-note}',
        content,
        flags=re.MULTILINE
    )

    return content


def process_file(filepath: Path) -> bool:
    """
    Process a single file and return True if changes were made.
    """
    try:
        original_content = filepath.read_text(encoding='utf-8')
        new_content = convert_margin_to_callout(original_content)

        if new_content != original_content:
            filepath.write_text(new_content, encoding='utf-8')
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def main():
    """Find and convert all column-margin blocks in Quarto documents."""
    lecture_note_dir = Path(__file__).parent / "docs" / "lecture-note"

    # Find all .qmd files
    qmd_files = list(lecture_note_dir.rglob("*.qmd"))

    print(f"Found {len(qmd_files)} .qmd files")

    modified_count = 0
    modified_files = []

    for qmd_file in qmd_files:
        if process_file(qmd_file):
            modified_count += 1
            modified_files.append(qmd_file.relative_to(lecture_note_dir))
            print(f"✓ Modified: {qmd_file.relative_to(lecture_note_dir)}")

    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"Modified {modified_count} file(s)")

    if modified_files:
        print(f"\nModified files:")
        for f in modified_files:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
