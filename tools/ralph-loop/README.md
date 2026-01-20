# Ralph Loop - Task Generation Tool

This tool generates a features.json file from task_instructions.json, creating atomic tasks for each file that needs to be processed.

## Features

- Support for explicit file paths
- Support for regex patterns to match multiple files
- Automatic priority assignment for pattern-matched files
- Backward compatible with existing configurations

## Usage

```bash
python generate_features.py
```

This reads `task_instructions.json` and generates `features.json`.

## File Specification Formats

### 1. Explicit Path (Backward Compatible)

Specify individual files explicitly:

```json
{
  "files": [
    {
      "path": "docs/lecture-note/m05-images/overview.qmd",
      "category": "m05-images",
      "priority": 39
    },
    {
      "path": "docs/lecture-note/m05-images/01-what-is-an-image.qmd",
      "category": "m05-images",
      "priority": 40
    }
  ]
}
```

### 2. Regex Pattern (New Feature)

Match multiple files with a regex pattern:

```json
{
  "files": [
    {
      "pattern": "m05-images/.*\\.qmd$",
      "category": "m05-images",
      "priority": 39
    }
  ]
}
```

This will find all `.qmd` files in the `m05-images` directory.

### Mixed Usage

You can combine both approaches:

```json
{
  "files": [
    {
      "pattern": "m04-text/.*\\.qmd$",
      "category": "m04-text",
      "priority": 1
    },
    {
      "path": "docs/lecture-note/special-file.qmd",
      "category": "special",
      "priority": 100
    }
  ]
}
```

## Pattern Syntax

The `pattern` field accepts Python regex patterns. The pattern is matched against file paths relative to the `projectDir` specified in `meta.projectDir`.

### Common Pattern Examples

- `m05-images/.*\\.qmd$` - All .qmd files in m05-images directory
- `m04-text/\\d+-.*\\.qmd$` - All .qmd files starting with digits in m04-text
- `.*overview\\.qmd$` - All overview.qmd files anywhere
- `(m04-text|m05-images)/.*\\.qmd$` - All .qmd files in either directory

### Pattern Matching Behavior

- Patterns are matched against the full relative path from `projectDir`
- Only files (not directories) are matched
- Matched files are sorted alphabetically
- Priority is auto-incremented for each matched file (priority, priority+1, priority+2, etc.)

## How It Works

1. The script reads `task_instructions.json`
2. For each file entry:
   - If `pattern` is specified, it expands to all matching files in `projectDir`
   - If `path` is specified, it uses the explicit path
3. Creates one feature per file with unique ID, priority, and task steps
4. Writes all features to `features.json`

## Example Output

For a pattern that matches 3 files with priority 10:

```json
{
  "features": [
    {
      "id": "m05-images-010-overview",
      "priority": 10,
      "category": "m05-images",
      "file": "m05-images/overview.qmd"
    },
    {
      "id": "m05-images-011-01-what-is-an-image",
      "priority": 11,
      "category": "m05-images",
      "file": "m05-images/01-what-is-an-image.qmd"
    },
    {
      "id": "m05-images-012-02-deep-learning",
      "priority": 12,
      "category": "m05-images",
      "file": "m05-images/02-deep-learning.qmd"
    }
  ]
}
```
