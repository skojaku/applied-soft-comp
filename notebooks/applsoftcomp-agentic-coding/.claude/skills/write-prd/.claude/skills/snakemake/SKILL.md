# Snakemake Style Guide

Copy `utils.smk` from `.claude/skills/snakemake/utils.smk` to project root for new projects.

## Project Layout

```
project/
├── Snakefile                    # Config, globals, params, includes
├── utils.smk                    # Utility functions (from this skill)
├── workflow/
│   ├── config.template.yaml     # Template (users copy to config.yaml)
│   ├── rules/*.smk              # One file per workflow stage
│   ├── <category>/              # Scripts grouped by purpose (stats/, plot/, fitting/)
│   └── preprocessing/{dataset}/ # Dataset-specific ingestion with own Snakefile
└── data/{dataset}/
    ├── preprocessed/            # Raw/cleaned inputs
    ├── derived/                 # Computed outputs
    └── plot_data/               # Aggregated data for figures
```

## Naming Conventions

| Thing | Convention | Example |
|-------|-----------|---------|
| File path constants | `UPPER_CASE` | `MODEL_FILE`, `FIG_DEGREE_DIST` |
| Parameter dicts | `lower_case` | `params_my_model` |
| Paramspace objects | `lower_case` | `my_paramspace` |
| Rule names | `snake_case` | `calc_degree_distribution` |
| Dataset/constant lists | `UPPER_CASE` | `DATA_LIST` |

## Snakefile Structure

```python
from os.path import join as j
include: "./utils.smk"
configfile: "workflow/config.yaml"

# Constants (UPPER_CASE)
DATA_DIR = config["data_dir"]
DATA_LIST = ["dataset_a", "dataset_b"]

# Parameter dicts (values ALWAYS lists)
params_my_model = {"dim": [64, 128], "c0": [5, 20]}

# Paramspace from param dicts
my_paramspace = to_paramspace(params_my_model)

# File path templates (use f-string + wildcard_pattern for parameterized paths)
INPUT_NET = j(DATA_DIR, "{data}", "preprocessed", "net.npz")
MODEL_FILE = j(DATA_DIR, "{data}", "derived", f"model_{my_paramspace.wildcard_pattern}.pt")
# -> data/{data}/derived/model_dim~{dim}_c0~{c0}.pt

include: "workflow/rules/stage_a.smk"

rule all:
    input:
        rules.stage_a_all.input,
```

## utils.smk Functions

- **`to_paramspace(dict_or_list)`** — Creates `Paramspace` from param dict(s). Use `.wildcard_pattern` in file paths.
- **`expand(filename, *param_dicts, **params)`** — Custom expand with partial_format support. Unfilled wildcards remain as `{name}`. **Always use this instead of Snakemake's built-in.**
- **`partial_format(template, **kwargs)`** — Fill some wildcards, leave the rest.
- **`make_filename(prefix, ext, names)`** — Build `prefix_key={key}.ext` templates.

## Writing Rules

```python
rule calc_something:
    input:
        net_file = INPUT_NET,          # Always NAMED inputs/outputs
    output:
        output_file = OUTPUT_FILE
    params:
        dim = lambda wildcards: wildcards.dim,  # Forward wildcards as params
        label = lambda wildcards: "A" if wildcards.data == "x" else "B",
    resources:
        gpu = 1
    threads: 4
    script:
        "../scripts_dir/calc-something.py"  # Relative to .smk file
```

Each `.smk` file has a master aggregation rule:

```python
rule stage_a_all:
    input:
        expand(OUTPUT_FILE, params_my_model, data=DATA_LIST),
```

## Rule Reuse

```python
use rule calc_stat as calc_stat_simulated with:
    input:
        net_file = SIMULATED_NET,
    output:
        output_file = SIMULATED_STAT
```

Inherits `script:`, `threads:`, `resources:` unless overridden.

## Script Convention

Scripts must work both in Snakemake and interactively:

```python
import sys
if "snakemake" in sys.modules:
    input_file = snakemake.input["input_file"]
    dim = int(snakemake.params["dim"])           # Params are STRINGS
    use_feature = snakemake.params["use_feature"] == "True"
else:
    input_file = "../data/sample.npz"
    dim = 64
    use_feature = True
```

Type conversions: `bool` → `== "True"`, `int` → `int()`, `float` → `float()`, `list` → as-is, `None` → `is None`.

## Key Rules

- Use `j()` for all paths; `to_paramspace()` + `.wildcard_pattern` for parameterized paths
- Double braces `{{wildcard}}` when embedding wildcards inside f-strings
- Parameter dict values are always lists, even single values
- Use custom `expand()` from utils.smk, not Snakemake's built-in
- Named (not positional) rule inputs/outputs
