# Context Engineering Diagram — Layout Outline

## Canvas Overview

- **Canvas size:** 1200 × 1600 px (logical units)
- **Left margin:** 20px (band labels)
- **Content area:** x=80 to x=1180 (width=1100)
- **Layout:** 7 horizontal bands stacked top-to-bottom
- **Band separator:** dashed horizontal lines (strokeStyle: dashed, roughness:0, roundness:null, backgroundColor: transparent, strokeColor: #aaaaaa)

---

## Band 0 — Title Bar (y: 0–80)

| Element | Type | Position | Size | Style |
|---------|------|----------|------|-------|
| Title text "Context Engineering" | text (standalone) | x=80, y=10 | fontSize:28, strokeColor:#1c1c1c |
| Subtitle text "Deliberate curation of what the LLM sees at each step of agent execution" | text (standalone) | x=80, y=48 | fontSize:14, strokeColor:#555555 |

---

## Band 1 — WHY (y: 80–320)

**Color:** background #ffc9c9, stroke #e03131
**Band label:** "WHY CONTEXT ENGINEERING?" — left edge, fontSize:12, uppercase, #e03131, rotated 90° or vertical text at x=20, y=200

### Header box
- Rectangle + text: "Why Context Engineering?" (fontSize:16, bold feeling via size)
- Position: x=80, y=90, width=1100, height=40
- backgroundColor: #ffc9c9, strokeColor: #e03131, strokeWidth:2

### Left problem box (Long context degrades performance)
- Rectangle: x=80, y=145, width=520, height=160
- backgroundColor: #ffc9c9, strokeColor: #e03131, strokeWidth:2, roundness:{type:3}
- Text (title): "Long Context Degrades Performance" (fontSize:14)
- Text (body): "LLM accuracy and instruction-following decline as the context window fills. More tokens ≠ better results." (fontSize:12)
- Combined text element inside box

### Right problem box (Lost in the middle)
- Rectangle: x=620, y=145, width=560, height=160
- backgroundColor: #ffc9c9, strokeColor: #e03131, strokeWidth:2, roundness:{type:3}
- Text (title): "Lost in the Middle" (fontSize:14)
- Text (body): "Attention weakens for information buried in the middle of a long context. Position matters — beginning and end are privileged." (fontSize:12)
- Combined text element inside box

### Takeaway text (bottom of band)
- Standalone text: "You cannot just dump everything into the context and hope for the best. Deliberate curation is required."
- Position: x=80, y=318, fontSize:12, strokeColor:#888888, italic style via color

---

## Band 2 — WRITE (y: 320–530)

**Color:** background #d0bfff, stroke #7048e8
**Band label:** "WRITE" — left edge x=20, y=425, fontSize:12, strokeColor:#7048e8

### Header box
- Rectangle + text: "WRITE — Save Outside the Window"
- x=80, y=330, width=1100, height=40
- backgroundColor: #d0bfff, strokeColor: #7048e8, strokeWidth:2

### Concept box (left column, ~40% width)
- Rectangle: x=80, y=385, width=430, height=130
- backgroundColor: #e5daff (lighter tint), strokeColor: #7048e8, strokeWidth:1, roundness:{type:3}
- Text: "Persist information outside the context window before it's lost or too expensive to keep. Read it back selectively later."
- fontSize:12

### Arrow (concept → examples)
- Elbow arrow from right edge of concept box to left edge of example area
- startPoint: (510, 450), endPoint: (540, 450)
- strokeColor: #7048e8, endArrowhead: "arrow"

### Example box 1 (Scratchpad)
- Rectangle: x=540, y=385, width=300, height=60
- backgroundColor: #e5daff, strokeColor: #7048e8, strokeWidth:1, roundness:{type:3}
- Text title: "Scratchpad" (fontSize:13, bold tint)
- Text body: "In the Ralph loop, progress.txt stores what each iteration accomplished. Next agent reads it — never re-reads all prior outputs." (fontSize:11)

### Example box 2 (Long-term memory)
- Rectangle: x=540, y=455, width=300, height=60
- backgroundColor: #e5daff, strokeColor: #7048e8, strokeWidth:1, roundness:{type:3}
- Text title: "Long-term Memory" (fontSize:13)
- Text body: "Lessons learned, user preferences, domain facts persist across sessions as external files or databases." (fontSize:11)

---

## Band 3 — SELECT (y: 530–740)

**Color:** background #b2f2bb, stroke #2f9e44
**Band label:** "SELECT" — left edge x=20, y=635, fontSize:12, strokeColor:#2f9e44

### Header box
- Rectangle + text: "SELECT — Pull In What's Relevant"
- x=80, y=540, width=1100, height=40
- backgroundColor: #b2f2bb, strokeColor: #2f9e44, strokeWidth:2

### Concept box (left column)
- Rectangle: x=80, y=595, width=430, height=130
- backgroundColor: #d3f9d8 (lighter tint), strokeColor: #2f9e44, strokeWidth:1, roundness:{type:3}
- Text: "Don't preload everything. Retrieve only what this specific step needs — keeping the context lean and focused."
- fontSize:12

### Arrow (concept → examples)
- Elbow arrow from right edge of concept box to left edge of example area
- strokeColor: #2f9e44, endArrowhead: "arrow"

### Example box 1 (RAG)
- Rectangle: x=540, y=595, width=300, height=60
- backgroundColor: #d3f9d8, strokeColor: #2f9e44, strokeWidth:1, roundness:{type:3}
- Text title: "RAG" (fontSize:13)
- Text body: "Instead of 50 documents, retrieve the 2-3 most relevant chunks at query time." (fontSize:11)

### Example box 2 (Tool results)
- Rectangle: x=540, y=665, width=300, height=60
- backgroundColor: #d3f9d8, strokeColor: #2f9e44, strokeWidth:1, roundness:{type:3}
- Text title: "Tool Results" (fontSize:13)
- Text body: "Only the output of the tool called right now enters context — not the full history of all prior calls." (fontSize:11)

---

## Band 4 — COMPRESS (y: 740–950)

**Color:** background #ffec99, stroke #f08c00
**Band label:** "COMPRESS" — left edge x=20, y=845, fontSize:12, strokeColor:#f08c00

### Header box
- Rectangle + text: "COMPRESS — Shrink What's Inside"
- x=80, y=750, width=1100, height=40
- backgroundColor: #ffec99, strokeColor: #f08c00, strokeWidth:2

### Concept box (left column)
- Rectangle: x=80, y=805, width=430, height=130
- backgroundColor: #fff3bf (lighter tint), strokeColor: #f08c00, strokeWidth:1, roundness:{type:3}
- Text: "Reduce the token footprint of content already in the context. Keep the meaning, discard the bulk."
- fontSize:12

### Arrow (concept → examples)
- Elbow arrow from right edge of concept box to left edge of example area
- strokeColor: #f08c00, endArrowhead: "arrow"

### Example box 1 (Summarize history)
- Rectangle: x=540, y=805, width=300, height=60
- backgroundColor: #fff3bf, strokeColor: #f08c00, strokeWidth:1, roundness:{type:3}
- Text title: "Summarize History" (fontSize:13)
- Text body: "Replace a 10-turn conversation transcript with a 3-sentence summary before the next request." (fontSize:11)

### Example box 2 (Prune tool logs)
- Rectangle: x=540, y=875, width=300, height=60
- backgroundColor: #fff3bf, strokeColor: #f08c00, strokeWidth:1, roundness:{type:3}
- Text title: "Prune Tool Logs" (fontSize:13)
- Text body: "Remove intermediate reasoning steps and redundant tool call echoes that no longer inform the next action." (fontSize:11)

---

## Band 5 — ISOLATE (y: 950–1180)

**Color:** background #99e9f2, stroke #0c8599
**Band label:** "ISOLATE" — left edge x=20, y=1065, fontSize:12, strokeColor:#0c8599

### Header box
- Rectangle + text: "ISOLATE — Split Across Agents"
- x=80, y=960, width=1100, height=40
- backgroundColor: #99e9f2, strokeColor: #0c8599, strokeWidth:2

### Concept box (left column)
- Rectangle: x=80, y=1015, width=430, height=150
- backgroundColor: #c5f6fa (lighter tint), strokeColor: #0c8599, strokeWidth:1, roundness:{type:3}
- Text: "Decompose a long task into sub-tasks, each handled by a separate agent with its own focused context window. No single agent sees everything."
- fontSize:12

### Arrow (concept → examples)
- Elbow arrow from right edge of concept box to left edge of example area
- strokeColor: #0c8599, endArrowhead: "arrow"

### Example box 1 (Ralph loop)
- Rectangle: x=540, y=1015, width=300, height=70
- backgroundColor: #c5f6fa, strokeColor: #0c8599, strokeWidth:1, roundness:{type:3}
- Text title: "Ralph Loop" (fontSize:13)
- Text body: "Each story runs in its own isolated agent. Agent for US-003 sees only prompt.md + prd.json — not history of US-001 and US-002." (fontSize:11)

### Example box 2 (80/20 rule)
- Rectangle: x=540, y=1095, width=300, height=70
- backgroundColor: #c5f6fa, strokeColor: #0c8599, strokeWidth:1, roundness:{type:3}
- Text title: "80/20 Rule: Plan + Execute" (fontSize:13)
- Text body: "Planner agent reads broadly to produce a plan. Executor agent sees only the plan + current task — keeping execution context tight." (fontSize:11)

---

## Band 6 — Summary Row (y: 1180–1340)

**Color:** background #f8f9fa, stroke #aaaaaa
**Band label:** "SUMMARY" — left edge x=20, y=1260, fontSize:12, strokeColor:#aaaaaa

### Summary header text
- Standalone text: "The Four Strategies at a Glance"
- x=80, y=1190, fontSize:16, strokeColor:#333333

### Four summary boxes (equal width, side by side)

Each box: width=256, height=120, gap=12
Starting x=80, y=1215

**WRITE summary box**
- x=80, y=1215, width=256, height=120
- backgroundColor: #d0bfff, strokeColor: #7048e8, strokeWidth:2, roundness:{type:3}
- Text: "WRITE\nSave outside the window.\nUse scratchpads & long-term memory."

**SELECT summary box**
- x=348, y=1215, width=256, height=120
- backgroundColor: #b2f2bb, strokeColor: #2f9e44, strokeWidth:2, roundness:{type:3}
- Text: "SELECT\nPull in when needed.\nUse RAG & targeted retrieval."

**COMPRESS summary box**
- x=616, y=1215, width=256, height=120
- backgroundColor: #ffec99, strokeColor: #f08c00, strokeWidth:2, roundness:{type:3}
- Text: "COMPRESS\nShrink what's inside.\nSummarize & prune redundancy."

**ISOLATE summary box**
- x=884, y=1215, width=256, height=120
- backgroundColor: #99e9f2, strokeColor: #0c8599, strokeWidth:2, roundness:{type:3}
- Text: "ISOLATE\nSplit across agents.\nEach window stays focused."

---

## Band Separators (dashed horizontal lines)

| After Band | y position | Description |
|------------|------------|-------------|
| Band 0 | y=80 | thin dashed rect, height=2, full width |
| Band 1 | y=320 | thin dashed rect |
| Band 2 | y=530 | thin dashed rect |
| Band 3 | y=740 | thin dashed rect |
| Band 4 | y=950 | thin dashed rect |
| Band 5 | y=1180 | thin dashed rect |

Each separator:
- x=0, width=1200, height=2
- strokeColor: #aaaaaa, backgroundColor: transparent
- strokeStyle: dashed, roughness:0, roundness:null

---

## Element ID Convention

Use descriptive snake_case IDs:
- `band1_header`, `band1_header_text`
- `band1_left_box`, `band1_left_text`
- `band1_right_box`, `band1_right_text`
- `band2_header`, `band2_header_text`
- `band2_concept_box`, `band2_concept_text`
- `band2_arrow`, `band2_ex1_box`, `band2_ex1_text`, `band2_ex2_box`, `band2_ex2_text`
- (same pattern for bands 3–5)
- `band6_write_box`, `band6_write_text`, etc.
- `sep_0`, `sep_1`, `sep_2`, `sep_3`, `sep_4`, `sep_5`

---

## Relationships & Arrows Summary

| Arrow ID | From | To | Direction | Color |
|----------|------|----|-----------|-------|
| band2_arrow | band2_concept_box (right edge) | band2_ex1_box (left edge) | horizontal right | #7048e8 |
| band3_arrow | band3_concept_box (right edge) | band3_ex1_box (left edge) | horizontal right | #2f9e44 |
| band4_arrow | band4_concept_box (right edge) | band4_ex1_box (left edge) | horizontal right | #f08c00 |
| band5_arrow | band5_concept_box (right edge) | band5_ex1_box (left edge) | horizontal right | #0c8599 |

---

## Teaching Flow

The diagram reads **top-to-bottom** like a story:

1. **Band 0 (Title):** Orients the student — "This is about context engineering."
2. **Band 1 (WHY):** Establishes urgency — two hard problems (degradation + lost-in-middle) make context curation necessary.
3. **Bands 2–5 (Strategies):** Each band addresses a strategy. Left column explains the concept; right column grounds it in concrete examples from the Ralph Wiggum loop and real systems.
4. **Band 6 (Summary):** Four compact boxes reinforce the four strategies at a glance — useful as a quick reference.
