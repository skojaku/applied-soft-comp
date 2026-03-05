# Review v4 — diagram_v4.png

**Issues found: yes**

## Summary

Major improvement over v3 — the minimalistic design is effective. White backgrounds with colored borders give the diagram a clean, reference-sheet feel. The band structure is clearly readable. However, two categories of issues remain.

## Specific Issues

1. **Strategy bands — concept text overlaps header** — In the WRITE, SELECT, COMPRESS, and ISOLATE bands, the concept description text (left column, standalone text element at x=12) visually appears to bleed into the header bar. The Virgil hand-drawn font renders with upward ascenders that bridge the visual gap between concept text and header bottom. The gap is ~38px which is geometrically correct, but the visual effect is that the concept text starts too close to the header. Two options: (a) increase gap to 50px or (b) give concept text a white background so it visually separates from the header.

2. **WHY band — title text overlaps body text** — The title text at y=140 and body text at y=172 are 32px apart. With Virgil font at fontSize=12 (≈15px rendered height), there is a 17px gap which is too tight for the hand-drawn style. The title of the left box ("Long Context Degrades Performance") and the body text visually merge. Increase gap to at least 25px.

3. **WHY band — cross-box text bleeding** — The left title text (width=368 ending at x=382) and right title text (starting at x=418) have a 36px gap, which is adequate. However the left body text and right title text appear at similar y values and could create visual noise near the center of the canvas.

4. **Minimalistic design — success** — The white box backgrounds with colored stroke borders (replacing the filled colored backgrounds) significantly improves readability and reduces visual clutter. This is a clear improvement that addresses the user's feedback. The summary row at the bottom is especially clean.

5. **Band separation — good** — The dashed separator lines between bands are clearly visible. Color-coded headers with subtle tint (opacity:40) effectively distinguish sections without being heavy.

6. **Example boxes — clean** — The right-column example boxes (Scratchpad, Long-term Memory, RAG, Tool Results, Summarize History, Prune Tool Logs, Ralph Loop, 80/20 Rule) all render cleanly with white backgrounds and readable text.

7. **Overall narrative flow** — A student can follow the story from WHY → WRITE → SELECT → COMPRESS → ISOLATE → Summary. Color coding consistently identifies each strategy.

## Required Fixes for v5

1. Increase gap between each strategy band header bottom and concept_box top from 30px to 50px (shift concept/example boxes down another 20px per band, cascading through all subsequent bands).
2. In the WHY band boxes, increase gap between title text bottom (y=140+15=155) and body text top from 17px to 25px: move body text from y=172 to y=180.
3. Increase box heights in WHY band to accommodate the extra spacing.
4. Give the concept text elements a white `backgroundColor` (not transparent) so they visually separate from any nearby header rendering.
