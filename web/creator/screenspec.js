// The screen spec, mirrored: `creator/screens/spec.py` and the geometry of
// `creator/screens/tracker.shapes`. Pure — no DOM, no imports — so `state.js`
// can read and write a card's screens with it, and so
// `tests/test_screens_mirror.py` can load it in node and hold it to Python,
// which is authoritative. The UI that draws screens is `screens.js`.

// ---- the spec (mirrors creator/screens/spec.py) ----------------------------------

export const DEVICES = [
  { id: "phone", label: "Smartphone", noun: "phone screen", width: 1080, height: 2340 },
  { id: "tablet", label: "Tablet", noun: "tablet screen", width: 1536, height: 2048 },
  { id: "laptop", label: "Laptop", noun: "laptop screen", width: 1920, height: 1200 },
  { id: "monitor", label: "Desktop monitor", noun: "monitor screen", width: 1920, height: 1080 },
  { id: "tv", label: "TV", noun: "TV screen", width: 1920, height: 1080 },
  { id: "custom", label: "Custom", noun: "screen", width: 1920, height: 1080 },
];
export const DEFAULT_DEVICE = "phone";
export const CUSTOM_LONG_SIDE = 1920;
export const MAX_CUSTOM_RATIO = 4.0;

export const COLOURS = [
  { id: "white", label: "White", fill: [242, 242, 242], hue: null },
  { id: "magenta", label: "Magenta", fill: [255, 0, 255], hue: 300.0 },
  { id: "green", label: "Green", fill: [0, 255, 0], hue: 120.0 },
  { id: "blue", label: "Blue", fill: [0, 0, 255], hue: 240.0 },
];
export const PATTERNS = ["grid", "plus", "none"];
export const DEFAULT_STYLE = { pattern: "none", colour: "green" };

export const MAX_SCREENS = 2;
export const FITS = ["fill", "fit", "stretch"];
export const WHEN_SHORT = ["hold", "loop", "first"];
export const CUTS = ["continue", "restart"];
export const KINDS = ["image", "video"];
export const MAX_OFFSET_S = 3600.0;

export const deviceOf = (id) => DEVICES.find((device) => device.id === id) ?? null;
export const colourOf = (id) => COLOURS.find((colour) => colour.id === id) ?? null;

/** A settings value -> `{pattern, colour}`, falling back to the default for
 *  anything this build does not know. Mirrors `spec.clean_style`, leniently:
 *  the server refuses what this forgives. */
export function cleanStyle(raw) {
  const style = raw && typeof raw === "object" ? raw : {};
  return {
    pattern: PATTERNS.includes(style.pattern) ? style.pattern : DEFAULT_STYLE.pattern,
    colour: colourOf(style.colour) ? style.colour : DEFAULT_STYLE.colour,
  };
}

/** The key colour of each of `count` screens. Mirrors `spec.assign_colours`. */
export function assignColours(first, count) {
  const order = [first, ...COLOURS.map((c) => c.id).filter((id) => id !== first)];
  return order.slice(0, count);
}

/** The tracker's size for a screen, or null for a custom aspect the compiler
 *  would refuse. Mirrors `spec._size`, Python's rounding included: `round`
 *  there is half-to-even, which is what `roundHalfEven` reproduces. */
export function trackerSize(screen) {
  const device = deviceOf(screen?.device ?? DEFAULT_DEVICE);
  if (!device) return null;
  if (device.id !== "custom") return { width: device.width, height: device.height };
  const aspect = screen.aspect;
  if (!Array.isArray(aspect) || aspect.length !== 2
      || !aspect.every((v) => typeof v === "number" && Number.isFinite(v) && v > 0)) return null;
  const [w, h] = aspect;
  if (Math.max(w, h) / Math.min(w, h) > MAX_CUSTOM_RATIO) return null;
  const scale = CUSTOM_LONG_SIDE / Math.max(w, h);
  return { width: Math.max(2, roundHalfEven((w * scale) / 2) * 2),
           height: Math.max(2, roundHalfEven((h * scale) / 2) * 2) };
}

function roundHalfEven(x) {
  const floor = Math.floor(x);
  const diff = x - floor;
  if (Math.abs(diff - 0.5) > 1e-9) return Math.round(x);
  return floor % 2 === 0 ? floor : floor + 1;
}

/** What is wrong with screen `index` of a card, in the compiler's words, or
 *  null. Mirrors `spec.parse` screen by screen. */
export function screenProblem(screen, index) {
  const where = `screen ${index + 1}`;
  if (!screen || typeof screen !== "object") return `${where} must be an object`;
  if (!deviceOf(screen.device ?? DEFAULT_DEVICE)) {
    return `${where}'s device must be one of ${DEVICES.map((d) => d.id).join(", ")}, not ${JSON.stringify(screen.device)}`;
  }
  if ((screen.device ?? DEFAULT_DEVICE) === "custom" && !trackerSize(screen)) {
    const aspect = screen.aspect;
    const wellFormed = Array.isArray(aspect) && aspect.length === 2
      && aspect.every((v) => typeof v === "number" && Number.isFinite(v) && v > 0);
    return wellFormed
      ? `a custom screen can be at most ${MAX_CUSTOM_RATIO}:1 — the tracker's markers stop fitting past that`
      : "a custom screen needs its aspect as two positive numbers, width and height";
  }
  const content = screen.content;
  if (!content || typeof content !== "object" || typeof content.filename !== "string"
      || !content.filename.trim()) {
    return `${where} has nothing to show — pick a picture or a clip for it`;
  }
  return null;
}

/** A card's `screens`, as the editor keeps them: the well-formed entries, each
 *  a fresh object. The compiler is what refuses a bad one; this only keeps
 *  junk out of the editor's hands. */
export function parseScreens(raw) {
  if (!Array.isArray(raw)) return [];
  return raw.filter((screen) => screen && typeof screen === "object").map((screen) => {
    const out = { device: deviceOf(screen.device) ? screen.device : DEFAULT_DEVICE };
    if (screen.content && typeof screen.content === "object") {
      out.content = { filename: String(screen.content.filename ?? ""),
                      kind: KINDS.includes(screen.content.kind) ? screen.content.kind : "image" };
    }
    if (Array.isArray(screen.aspect)) out.aspect = screen.aspect.map(Number);
    if (FITS.includes(screen.fit)) out.fit = screen.fit;
    const offset = Number(screen.offset_s);
    if (Number.isFinite(offset) && offset > 0) out.offset_s = Math.min(offset, MAX_OFFSET_S);
    if (WHEN_SHORT.includes(screen.when_short)) out.when_short = screen.when_short;
    if (CUTS.includes(screen.cuts)) out.cuts = screen.cuts;
    return out;
  });
}

/** What a card writes about its screens: only what differs from the defaults,
 *  so a screen set up the obvious way is a device and a file. The tracker is
 *  never written — it is this machine's, stamped on by the node. */
export function serializeScreens(screens) {
  return (screens ?? []).map((screen) => ({
    device: screen.device ?? DEFAULT_DEVICE,
    ...(screen.device === "custom" && Array.isArray(screen.aspect)
      ? { aspect: screen.aspect.map(Number) } : {}),
    ...(screen.content ? { content: { filename: screen.content.filename,
                                      kind: screen.content.kind ?? "image" } } : {}),
    ...(screen.fit && screen.fit !== FITS[0] ? { fit: screen.fit } : {}),
    ...(Number(screen.offset_s) > 0 ? { offset_s: Math.round(Number(screen.offset_s) * 1000) / 1000 } : {}),
    ...(screen.when_short && screen.when_short !== WHEN_SHORT[0] ? { when_short: screen.when_short } : {}),
    ...(screen.cuts && screen.cuts !== CUTS[0] ? { cuts: screen.cuts } : {}),
  }));
}

// ---- the tracker (mirrors creator/screens/tracker.shapes) ---------------------------

const plusBars = (cx, cy, arm, stroke) => [
  [cx - arm, cy - stroke / 2, cx + arm, cy + stroke / 2],
  [cx - stroke / 2, cy - arm, cx + stroke / 2, cy + arm],
];

// Mirrors `tracker.GRID_SHORT` / `GRID_LONG`.
export const GRID_SHORT = 4;
export const GRID_LONG = [4, 9];

/** -> [columns, rows] of the grid pattern's squares. Mirrors `tracker.grid_size`,
 *  Python's half-to-even `round` included. */
export function gridSize(width, height) {
  const ratio = Math.max(width, height) / Math.min(width, height);
  const long = Math.min(GRID_LONG[1], Math.max(GRID_LONG[0], roundHalfEven(GRID_SHORT * ratio)));
  return width <= height ? [GRID_SHORT, long] : [long, GRID_SHORT];
}

/** `[[layer, kind, points]]` — see `tracker.shapes`. */
export function trackerShapes(pattern, width, height) {
  const s = Math.min(width, height);
  const out = [];
  if (pattern === "plus") {
    const inset = 0.09 * s;
    const arm = 0.045 * s;
    const stroke = Math.max(4.0, 0.012 * s);
    const corners = [[inset, inset], [width - inset, inset],
                     [width - inset, height - inset], [inset, height - inset]];
    const outline = 0.012 * s;
    const centre = [width / 2, height / 2];
    for (const [cx, cy] of corners) {
      for (const bar of plusBars(cx, cy, arm + 4, stroke + 8)) out.push(["rim", "rect", bar]);
    }
    for (const bar of plusBars(...centre, 0.14 * s + outline, 0.06 * s + 2 * outline)) {
      out.push(["rim", "rect", bar]);
    }
    for (const [cx, cy] of corners) {
      for (const bar of plusBars(cx, cy, arm, stroke)) out.push(["ink", "rect", bar]);
    }
    for (const bar of plusBars(...centre, 0.14 * s, 0.06 * s)) out.push(["ink", "rect", bar]);
  } else if (pattern === "grid") {
    const [cols, rows] = gridSize(width, height);
    const half = 0.035 * s;
    for (let i = 0; i < cols; i += 1) {
      for (let j = 0; j < rows; j += 1) {
        const cx = ((i + 1) / (cols + 1)) * width;
        const cy = ((j + 1) / (rows + 1)) * height;
        out.push(["ink", "rect", [cx - half, cy - half, cx + half, cy + half]]);
      }
    }
    const inset = 0.07 * s;
    const tri = 0.035 * s;
    for (const [cx, cy] of [[inset, inset], [width - inset, inset],
                            [width - inset, height - inset], [inset, height - inset]]) {
      out.push(["ink", "tri", [cx, cy, tri]]);
    }
  } else if (pattern !== "none") {
    throw new Error(`unknown tracker pattern ${JSON.stringify(pattern)}`);
  }
  return out;
}
