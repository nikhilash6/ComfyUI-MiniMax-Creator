"""Screen replacement: a tracker in the render, the real content in afterwards.

H3 draws screens well and draws the text on them badly unless it renders at
sizes nobody waits for. So the content a screen should show never goes through
the model at all. The model is handed a *tracker* as an ordinary reference
picture — a flat key colour with black markers — and told the screen shows it;
after decode, the screen pass finds that tracker in every frame, fits the
screen's four corners, and warps the real content onto them at whatever size
the reel leaves at. The text is the source file's text, pixel for pixel.

Four modules, split by what they may import:

- `spec` is pure: the screen types, the tracker palette and patterns, what a
  blob may say about a screen and the sentence the prompt carries. `compile.py`
  and `compile_image.py` read it before any loader exists, and the frontend
  mirrors it (`web/creator/screens.js`).
- `tracker` draws a tracker and its marker mask with numpy alone, and writes
  the PNG the model is handed into input/.
- `track` finds the screen in a frame: key, largest marked component, a
  four-sided fit, the shot cuts and the smoothing.
- `composite` puts the content on: warp, matte, shading, grain, despill.

`node.py` is the ComfyUI half — the pass over a finished reel and the node
between a still's decode and its save.
"""
