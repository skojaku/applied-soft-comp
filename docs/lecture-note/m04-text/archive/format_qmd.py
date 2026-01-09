import re
import black
from pathlib import Path

file_path = "/Users/skojaku-admin/Documents/projects/applied-soft-comp/docs/lecture-note/m03-text/semaxis.qmd"
content = Path(file_path).read_text()


def format_block(match):
    code = match.group(1)
    # Check if it's empty or just whitespace
    if not code.strip():
        return match.group(0)

    try:
        # Format with black
        formatted = black.format_str(code, mode=black.Mode())

        # Fix Quarto comments: # | -> #|
        formatted = re.sub(r"^# \| ", "#| ", formatted, flags=re.MULTILINE)

        # Ensure proper wrapping
        return f"```{{python}}\n{formatted}```"
    except Exception as e:
        print(f"Error formatting block: {e}")
        return match.group(0)


# Regex to match ```{python} ... ``` blocks
pattern = re.compile(r"```\{python\}(.*?)```", re.DOTALL)

new_content = pattern.sub(format_block, content)

Path(file_path).write_text(new_content)
print("Formatting complete.")
