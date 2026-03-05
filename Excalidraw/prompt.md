# Ralph Agent: Context Engineering Diagram

You are building the Excalidraw teaching diagram described in `prd.json`.

**Complete exactly ONE user story per run. Stop after one story.**

---

## Protocol

1. Read `prd.json`. Find the first story where `passes: false`.
2. Read `progress.txt` for context from prior iterations.
3. Read `user_input.txt` for corrections or instructions from the user — these take priority over everything else.
4. Implement that one story.
5. `git add -A`, `git commit -m "feat: [ID] - short title"`, `git push`.
6. Set `passes: true` on that story in `prd.json` and commit that change too.
7. Append one line to `progress.txt`: date, story ID, one-sentence summary of what was done.
8. Clear `user_input.txt` (overwrite with empty content).
9. **Stop. Do not implement further stories.**

If all stories have `passes: true`, output:
```
<promise>COMPLETE</promise>
```

---

## Diagram Content

This diagram teaches students about **Context Engineering** — the practice of strategically managing what information an LLM sees at each step of agent execution.

**Three-zone vertical layout:**

**Zone 1 — What fills the context window (top, y~150–480):**
- Title: "Context Engineering" (fontSize 28, centered)
- Subtitle: "Curating what goes into the LLM context window at every step of agent execution" (fontSize 14, gray)
- Three context-type boxes on the LEFT, each with colored arrows flowing RIGHT into the central context window:
  - **Instructions** (purple #d0bfff/#7048e8): prompts, examples, tool descriptions
  - **Knowledge** (green #b2f2bb/#2f9e44): facts, domain context, relevant info
  - **Tools** (yellow/orange #ffec99/#f08c00): API calls, results, feedback
- **LLM Context Window** (large blue box, #a5d8ff/#1971c2, strokeWidth:3) in the center-right:
  - Label: "LLM Context Window\n≈ RAM for Agents"
  - This is the focal element — make it visually prominent

**Zone 2 — The problem (middle, y~395–490):**
- **Context Poisoning** (red #ffc9c9/#e03131) directly below the context window:
  - Label: "⚠ Context Poisoning\nTokens accumulate → confusion,\nconflicts, degraded agents"
  - Red arrow straight down from context window bottom to this box

**Zone 3 — The strategies (bottom, y~500–730):**
- Dashed teal grouping rectangle labeled "4 STRATEGIES TO ENGINEER CONTEXT"
- Four equal teal boxes (#99e9f2/#0c8599) side by side:
  - **WRITE**: "Save outside context window / → scratchpad (session) / → memory (long-term)"
  - **SELECT**: "Pull in when needed / → RAG & retrieval / → semantic search"
  - **COMPRESS**: "Reduce token count / → summarize history / → prune old context"
  - **ISOLATE**: "Split across agents / → subagents / → specialized windows"
- Gray arrow from problem box bottom pointing down into strategies group

---

## Excalidraw Technical Rules

Every generated `.excalidraw` file must follow these rules exactly.

### File Structure
```json
{
  "type": "excalidraw",
  "version": 2,
  "source": "ralph-agent",
  "elements": [],
  "appState": { "gridSize": 20, "viewBackgroundColor": "#ffffff" },
  "files": {}
}
```

### Labels Require TWO Elements
The `label` property does not work in raw JSON. Every labeled shape needs two elements:

```json
// 1. Shape with boundElements
{
  "id": "my-box",
  "type": "rectangle",
  "boundElements": [{ "type": "text", "id": "my-box-text" }],
  "x": 0, "y": 0, "width": 200, "height": 80,
  "angle": 0, "strokeColor": "#1971c2", "backgroundColor": "#a5d8ff",
  "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid",
  "roughness": 1, "opacity": 100, "groupIds": [], "frameId": null,
  "roundness": { "type": 3 }, "seed": 1, "version": 1, "versionNonce": 1,
  "isDeleted": false, "updated": 1, "link": null, "locked": false
}
// 2. Separate text element
{
  "id": "my-box-text",
  "type": "text",
  "containerId": "my-box",
  "text": "My Label",
  "textAlign": "center",
  "verticalAlign": "middle",
  "x": 5,
  "y": 20,
  "width": 190,
  "height": 40,
  "fontSize": 14, "fontFamily": 1, "lineHeight": 1.25,
  "angle": 0, "strokeColor": "#1971c2", "backgroundColor": "transparent",
  "fillStyle": "solid", "strokeWidth": 1, "strokeStyle": "solid",
  "roughness": 1, "opacity": 100, "groupIds": [], "frameId": null,
  "roundness": null, "seed": 2, "version": 1, "versionNonce": 1,
  "isDeleted": false, "boundElements": null, "updated": 1, "link": null, "locked": false,
  "originalText": "My Label"
}
```

Text positioning:
- `x` = shape.x + 5
- `width` = shape.width - 10
- `y` = shape.y + (shape.height - text.height) / 2
- Always `textAlign: "center"`, `verticalAlign: "middle"`

Approximate text heights (fontSize 13, lineHeight 1.25):
- 1 line ≈ 17px, 2 lines ≈ 33px, 3 lines ≈ 49px, 4 lines ≈ 65px

### Never Use Diamond Shapes
Use styled rectangles instead of `type: "diamond"`.

### Elbow Arrows
For 90-degree corner arrows:
```json
{ "roughness": 0, "roundness": null, "elbowed": true }
```

Also required on arrows:
```json
"startArrowhead": null,
"endArrowhead": "arrow"
```

### Arrow Positioning
Arrow `x,y` starts at the source shape's edge — not the center:
- Top edge: `(shape.x + shape.width/2, shape.y)`
- Bottom edge: `(shape.x + shape.width/2, shape.y + shape.height)`
- Left edge: `(shape.x, shape.y + shape.height/2)`
- Right edge: `(shape.x + shape.width, shape.y + shape.height/2)`

Arrow final point offset = `(targetEdge.x - arrow.x, targetEdge.y - arrow.y)`

Arrow `width` = max absolute x-offset in points.
Arrow `height` = max absolute y-offset in points.
(Use width=1 or height=1 as minimum for straight arrows.)

### Arrow Bindings
Use `startBinding` and `endBinding` to visually attach arrows to shapes:
```json
"startBinding": { "elementId": "source-id", "focus": 0, "gap": 1, "fixedPoint": [1, 0.5] },
"endBinding": { "elementId": "target-id", "focus": 0, "gap": 1, "fixedPoint": [0, 0.5] }
```

fixedPoint values: top-center=[0.5,0], bottom-center=[0.5,1], left-center=[0,0.5], right-center=[1,0.5]

Also add the arrow ID to each shape's `boundElements` array.

### Dashed Grouping Rectangles
```json
{
  "strokeStyle": "dashed",
  "roughness": 0,
  "roundness": null,
  "backgroundColor": "transparent",
  "boundElements": null
}
```

### Required Element Properties (every element must have all of these)
```json
{
  "id": "unique-string",
  "type": "rectangle",
  "x": 0, "y": 0, "width": 160, "height": 80,
  "angle": 0,
  "strokeColor": "#1971c2",
  "backgroundColor": "#a5d8ff",
  "fillStyle": "solid",
  "strokeWidth": 2,
  "strokeStyle": "solid",
  "roughness": 1,
  "opacity": 100,
  "groupIds": [],
  "frameId": null,
  "roundness": { "type": 3 },
  "seed": 1,
  "version": 1,
  "versionNonce": 1,
  "isDeleted": false,
  "boundElements": null,
  "updated": 1,
  "link": null,
  "locked": false
}
```

---

## PNG Export Procedure

Use Playwright MCP tools to export. Run this sequence:

### 1. Start HTTP server
```bash
python3 -m http.server 8765 &
SERVER_PID=$!
sleep 1
```

### 2. Navigate browser
```
browser_navigate → http://localhost:8765/
```

### 3. Export PNG (browser_run_code)
```javascript
async (page) => {
  const fs = require('fs');
  const excalidrawJson = fs.readFileSync('diagram_v1.excalidraw', 'utf8');
  const pngBase64 = await page.evaluate(async (json) => {
    const utils = await import('https://esm.sh/@excalidraw/utils@0.1.2');
    const { exportToBlob } = utils.default;
    const data = JSON.parse(json);
    const blob = await exportToBlob({
      elements: data.elements,
      appState: { ...data.appState, exportBackground: true },
      files: data.files || {},
      mimeType: 'image/png'
    });
    const reader = new FileReader();
    return new Promise((resolve) => {
      reader.onloadend = () => resolve(reader.result);
      reader.readAsDataURL(blob);
    });
  }, excalidrawJson);
  return pngBase64;
}
```

### 4. Decode and save
Strip the `data:image/png;base64,` prefix, then:
```bash
echo "BASE64_DATA_HERE" | base64 -d > diagram_v1.png
```

### 5. Clean up
```
browser_close
```
```bash
kill $SERVER_PID 2>/dev/null || true
```

---

## Diagram Review Criteria

When reviewing the exported PNG, check for:

| Issue | What to Look For |
|-------|-----------------|
| Overlapping text | Any text that visually overlaps another element |
| Disconnected arrows | Arrows that don't visually connect to shapes |
| Narrative clarity | Can a student trace: inputs → context window → problem → strategies? |
| Text size | Strategy titles (WRITE/SELECT/COMPRESS/ISOLATE) should stand out visually |
| Color coding | Context-type arrows should match their source box colors |
| Teaching effectiveness | Does the diagram stand alone as a teaching artifact without verbal explanation? |
| Simplification | Elements that add clutter without adding meaning |

Save findings as `review_v1.md` (or `review_v2.md`, etc.) with:
- A pass/fail summary line: "Issues found: yes/no"
- A numbered list of specific issues with descriptions
- If no issues: write "No issues found. Ready for final presentation."

---

## Versioning Convention

- Excalidraw files: `diagram_v1.excalidraw`, `diagram_v2.excalidraw`, ...
- PNG exports: `diagram_v1.png`, `diagram_v2.png`, ...
- Review files: `review_v1.md`, `review_v2.md`, ...
- Never overwrite previous versions.
