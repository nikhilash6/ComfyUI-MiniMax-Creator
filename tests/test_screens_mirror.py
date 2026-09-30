"""`screenspec.js` still agrees with `creator/screens/spec.py` and `tracker.shapes`.

    python3 tests/test_screens_mirror.py

The frontend keeps its own copy of what a screen may be — to draw the tool, the
device tiles and the tracker preview, and to say what is wrong before a render
is queued — and its own copy of the tracker's geometry, which the preview draws
and the model is then handed drawn by Python. Python is authoritative. Skips
itself if node is not installed.
"""

import json

import layout

layout.skip_without_node()

from harness import FAILURES, check, passed

passed("screenspec.js mirrors the screen spec, its sizes, its errors and the "
       "tracker's geometry")

_pkg = layout.load("screen_spec", "screen_tracker")
spec, tracker = _pkg.screen_spec, _pkg.screen_tracker

SIZES = [(1080, 2340), (1920, 1080), (1536, 2048), (1920, 822), (400, 400)]
ASPECTS = [[21, 9], [9, 16], [4, 3], [2.39, 1], [1, 1], [3.9, 1], [4.1, 1], [10, 1],
           [0, 1], [7, 5], [5, 3]]
SCREENS = [
    {"handle": "phone-1", "device": "phone", "content": {"filename": "a.png"}},
    {"handle": "watch-1", "device": "watch", "content": {"filename": "a.png"}},
    {"handle": "screen-1", "device": "custom", "content": {"filename": "a.png"}},
    {"handle": "screen-1", "device": "custom", "aspect": [10, 1], "content": {"filename": "a.png"}},
    {"handle": "screen-1", "device": "custom", "aspect": [16, 9], "content": {"filename": "a.png"}},
    {"handle": "tv-1", "device": "tv"},
    {"handle": "laptop-1", "device": "laptop", "content": {"filename": "  "}},
    {"device": "phone", "content": {"filename": "a.png"}},
    {"handle": "phone", "device": "phone", "content": {"filename": "a.png"}},
]

SCRIPT = """
const s = await import(process.argv[1]);
const sizes = JSON.parse(process.argv[2]);
const aspects = JSON.parse(process.argv[3]);
const screens = JSON.parse(process.argv[4]);
console.log(JSON.stringify({
  devices: s.DEVICES,
  colours: s.COLOURS.map((c) => ({ id: c.id, fill: c.fill, hue: c.hue })),
  patterns: s.PATTERNS, fits: s.FITS, whenShort: s.WHEN_SHORT, cuts: s.CUTS,
  maxScreens: s.MAX_SCREENS, defaultStyle: s.DEFAULT_STYLE,
  shapes: Object.fromEntries(s.PATTERNS.flatMap((p) =>
    sizes.map(([w, h]) => [`${p}:${w}x${h}`, s.trackerShapes(p, w, h)]))),
  custom: aspects.map((aspect) => s.trackerSize({ device: "custom", aspect })),
  assign: s.PATTERNS.length && s.COLOURS.map((c) => s.assignColours(c.id, 3)),
  problems: screens.map((screen, i) => s.screenProblem(screen, i)),
  roundtrip: s.serializeScreens(s.parseScreens([
    { handle: "screen-2", device: "custom", aspect: [21, 9],
      content: { filename: "b.mp4", kind: "video" },
      fit: "fit", offset_s: 1.25, when_short: "loop", cuts: "restart", junk: 1 },
    { device: "phone", content: { filename: "a.png" }, fit: "fill", when_short: "hold" },
    { device: "phone", content: { filename: "c.png" } },
    "not a screen",
  ])),
  next: [s.nextScreenHandle("laptop", new Set(["laptop-1", "phone-2"])),
         s.nextScreenHandle("custom", new Set())],
}));
"""

js = layout.run(SCRIPT, layout.js("screenspec.js"), SIZES, ASPECTS, SCREENS)

check("devices", [{k: d[k] for k in ("id", "label", "word", "width", "height")} for d in js["devices"]],
      [{"id": d.id, "label": d.label, "word": d.word, "width": d.width, "height": d.height}
       for d in spec.DEVICES])
check("colours", js["colours"],
      [{"id": c.id, "fill": list(c.fill), "hue": c.hue} for c in spec.COLOURS])
for name, got, want in (("patterns", js["patterns"], list(spec.PATTERNS)),
                        ("fits", js["fits"], list(spec.FITS)),
                        ("when short", js["whenShort"], list(spec.WHEN_SHORT)),
                        ("cuts", js["cuts"], list(spec.CUTS)),
                        ("max screens", js["maxScreens"], spec.MAX_SCREENS),
                        ("default style", js["defaultStyle"], spec.DEFAULT_STYLE)):
    check(name, got, want)

for pattern in spec.PATTERNS:
    for w, h in SIZES:
        want = [[layer, kind, list(points)] for layer, kind, points in tracker.shapes(pattern, w, h)]
        got = js["shapes"][f"{pattern}:{w}x{h}"]
        same = len(got) == len(want) and all(
            g[0] == p[0] and g[1] == p[1]
            and all(abs(a - b) < 1e-9 for a, b in zip(g[2], p[2]))
            for g, p in zip(got, want))
        check(f"{pattern} tracker at {w}x{h} is drawn from the same shapes", same, True)

for aspect, got in zip(ASPECTS, js["custom"]):
    try:
        (screen,) = spec.parse([{"handle": "screen-1", "device": "custom", "aspect": aspect,
                                 "content": {"filename": "a.png"}}])
        want = {"width": screen.width, "height": screen.height}
    except spec.ScreenError:
        want = None
    check(f"a custom {aspect[0]}:{aspect[1]} screen is the same size", got, want)

check("colours are assigned in the same order",
      js["assign"], [spec.assign_colours(c.id, 3) for c in spec.COLOURS])

for index, (screen, got) in enumerate(zip(SCREENS, js["problems"])):
    try:
        spec.parse([screen])
        want = None
    except spec.ScreenError as exc:
        want = str(exc)
    if screen.get("device") == "watch":
        # Python quotes the value with repr, JS with JSON; the sentence is the same.
        check("an unknown device is refused on both sides", (got is None, want is None), (False, False))
        continue
    check(f"screen case {index}: the same problem, in the same words",
          got, want.replace("screen 1", f"screen {index + 1}") if want else None)

check("a card's screens round-trip to only what differs from the defaults, a screen "
      "saved without a handle given the next free one for its device", js["roundtrip"], [
    {"handle": "screen-2", "device": "custom", "aspect": [21, 9],
     "content": {"filename": "b.mp4", "kind": "video"},
     "fit": "fit", "offset_s": 1.25, "when_short": "loop", "cuts": "restart"},
    {"handle": "phone-1", "device": "phone", "content": {"filename": "a.png", "kind": "image"}},
    {"handle": "phone-2", "device": "phone", "content": {"filename": "c.png", "kind": "image"}},
])
check("and what they write, Python reads as the screens they were",
      [(s.handle, s.device, s.fit, s.offset_s, s.when_short, s.cuts)
       for s in spec.parse(js["roundtrip"][:2])],
      [("screen-2", "custom", "fit", 1.25, "loop", "restart"),
       ("phone-1", "phone", "fill", 0.0, "hold", "continue")])
check("a new handle is the device's word and the first free number",
      js["next"], ["laptop-2", "screen-1"])
