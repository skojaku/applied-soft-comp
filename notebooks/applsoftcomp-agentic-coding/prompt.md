# Ralph Agent Instructions

You are implementing the marimo notebooks described in `prd.json`. Each run, complete ONE user story.

## Steps

1. Read `prd.json`. Find the first story where `passes: false`.
2. Read `progress.txt` for context from prior iterations.
3. Implement the story. All requirements are in `prd.json` under `shared_requirements`, `notebooks`, and the story itself.
4. `git add`, `git commit` (`feat: [ID] - short title`), `git push`.
5. Set `passes: true` on the story in `prd.json` and commit that too.
6. Append one line to `progress.txt`: date, story ID, and one sentence on what was done.

## Done?

If all stories have `passes: true`, output:

```
<promise>COMPLETE</promise>
```
