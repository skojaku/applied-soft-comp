# Review v2 — diagram_v2.png

**Issues found: yes**

## Summary
Major improvement over v1 — all text is now visible and the diagram conveys the full teaching narrative. The 5-band structure (WHY, WRITE, SELECT, COMPRESS, ISOLATE) plus summary row reads clearly top-to-bottom. However, several text overlap issues remain that reduce readability.

## Specific Issues

1. **Left concept box text overlaps band headers** — In each of the WRITE, SELECT, COMPRESS, and ISOLATE bands, the concept description text (left column) renders visually into the header bar area. This happens because the concept text is left-aligned and there is insufficient vertical gap between the header bottom and the concept box top (currently only ~8px). The text should not overlap the header.

2. **WHY band title texts overlap box content** — The bold title text ("Long Context Degrades Performance", "Lost in the Middle") at y=128 in each box overlaps the body text starting at y=150. With the hand-drawn font at fontSize=12, the title takes ~15px and the body starts 22px below — this is tight but okay. However, both left and right box texts overlap slightly in the center where neither box has a visible gap (left box ends at x=392, right box starts at x=408 — only 16px gap).

3. **Concept box text appears behind/inside header visually** — The rough-style rendering of both the header box and concept box with no visual gap between them makes the concept text appear to be part of the header. Increase vertical spacing by 20-30px between each header and its concept/example boxes.

4. **Right-column example boxes: text is well positioned** — The Scratchpad, Long-term Memory, RAG, Tool Results, Summarize History, Prune Tool Logs, Ralph Loop, and 80/20 Rule boxes all render cleanly with readable text. Good.

5. **Summary row is readable and well-positioned** — All four summary boxes (WRITE, SELECT, COMPRESS, ISOLATE) are clear with proper labels.

6. **Overall narrative flow** — A student can follow the top-to-bottom story: WHY (two problems) → four strategies → summary. This is effective.

## Required Fixes for v3
1. Add 20px gap between each band header bottom and the concept/example boxes top (shift all boxes down within each band).
2. Increase left box width slightly and use slightly smaller font for the concept description to ensure text fits without overflowing.
3. Adjust the WHY band so left box right edge (x=392) has at least 20px gap before right box start (x=408+) — or simply use a consistent 16px gap which is fine.
4. Make header text bold (bold is not supported directly but could use fontSize 15 instead of 14 for more visual weight).
