# Review v3 — diagram_v3.png

**Issues found: yes**

## Summary

Good improvement over v2 — the 20px gap added between headers and content boxes makes the structure cleaner in most bands. Text is visible throughout. However, several visual overlaps remain that reduce readability.

## Specific Issues

1. **WHY band — header text overlaps box content** — The centered header text "WHY — Context Windows Have Hard Limits" at y=88 sits inside the header bar (y=80–112). The two content boxes start at y=120, and their title texts begin at y=128 — only 16px below the header bottom. The hand-drawn font renders slightly taller than estimated, so the bold title text in each box visually merges with the header text above. Increase the gap between band1_header bottom (y=112) and band1_left_box top from 8px to at least 20px (move boxes to y=132+).

2. **Strategy band concept text overlaps header** — In the WRITE, SELECT, COMPRESS, and ISOLATE bands, the left-column concept text is a standalone text element that begins 8px below the concept box top. The concept box is positioned 20px below the header bottom. However, the concept text visually bleeds into the header bar because the left-column concept text renders without a background — the header bar background color bleeds through. The fix is to ensure the concept box has an opaque fill (already has) AND the concept text y is calculated correctly. Visually the overlap appears because the concept text x=12 and the header text is centered on the same 800px canvas — they share vertical screen space. Increase the gap to 30px between header bottom and concept_box top for strategy bands.

3. **WHY band — body text starts too early** — In band1_left_box, the title text at y=128 and body text at y=150 are only 22px apart. With rough-style font rendering, this is too tight. Increase to at least 30px separation between title and body within each WHY problem box.

4. **COMPRESS band — concept text overlaps header** — The "Reduce the token footprint..." text is visually overlapping the "COMPRESS — Shrink What's Inside" header text in the middle of the canvas. This is the most obvious overlap in the diagram.

5. **ISOLATE band — concept text overlaps header** — Same as COMPRESS. "Decompose a long task..." renders into the header bar area.

6. **Arrow positioning** — The arrows from concept boxes to example boxes are well-positioned and match colors correctly. No issues here.

7. **Summary row** — Clean and readable. All four strategy summary boxes (WRITE, SELECT, COMPRESS, ISOLATE) render with clear text and proper separation.

8. **Overall narrative flow** — A student can still follow the top-to-bottom story despite the overlaps. The color coding is clear and effective.

## Required Fixes for v4

1. Increase gap between all band headers and their content boxes from 20px to 30px (shift all concept/example box y values down another 10px per band, cascading through all subsequent bands).
2. In the WHY band (band 1), increase the gap from header bottom to box tops from 8px to 20px.
3. In the WHY band boxes, increase the gap between title text and body text from 22px to 32px (move body text y down by 10px within each box, and increase box height by 10px).
4. These fixes require cascading all downstream band y values accordingly.
