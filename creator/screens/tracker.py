"""Drawing a tracker, and knowing where its markers are.

The same drawing serves twice. It is the PNG the model is handed as a reference
(`ensure` writes it into input/), and it is what the screen pass follows: the
markers' centres (`anchors`) are the points a frame's marker holes are matched
to, and the markers' pixels (`markers`) what a fit is checked against.

numpy alone, no PIL, so the drawing and the tracking share one rasteriser and a
pure-Python test can check both. Geometry is relative to the short side `s`,
as the spec gives it.
"""

import os
import re

import numpy as np

from . import spec

INK = (0, 0, 0)
RIM = (255, 255, 255)


def _plus(cx, cy, arm, stroke):
    """A plus as its two bars, (x0, y0, x1, y1) each."""
    return [(cx - arm, cy - stroke / 2, cx + arm, cy + stroke / 2),
            (cx - stroke / 2, cy - arm, cx + stroke / 2, cy + arm)]


# How many squares the grid lays across the screen's short side. The long side
# gets as many as keep them about as far apart, within these bounds. Four and
# up to nine rather than the spec's three by three: a person walking in front
# of a screen hides half of it, and the tracker is followed by whichever
# markers are left — the more of them, the more are left. H3 drew the 3 x 3
# grid back exactly on both lab clips; the denser one is what a lab test is
# for.
GRID_SHORT = 4
GRID_LONG = (4, 9)


def grid_size(width, height):
    """-> (columns, rows) of the grid pattern's squares."""
    ratio = max(width, height) / min(width, height)
    long = int(min(GRID_LONG[1], max(GRID_LONG[0], round(GRID_SHORT * ratio))))
    return (GRID_SHORT, long) if width <= height else (long, GRID_SHORT)


def anchors(pattern, width, height):
    """The markers' centres in tracker pixels: `[(kind, x, y, half)]`.

    `kind` is "square", "tri", "cross" or "plus"; `half` is its half-width. The
    tracker is followed by matching what a frame shows to these, so they are
    the same geometry `shapes` draws, said as points. A triangle's centre here
    is its centroid, a third of the way up from its base — where the middle of
    the blob it leaves in a frame is.
    """
    s = min(width, height)
    out = []
    if pattern == "plus":
        inset = 0.09 * s
        for cx, cy in ((inset, inset), (width - inset, inset),
                       (width - inset, height - inset), (inset, height - inset)):
            out.append(("cross", cx, cy, 0.045 * s))
        out.append(("plus", width / 2, height / 2, 0.14 * s))
    elif pattern == "grid":
        cols, rows = grid_size(width, height)
        half = 0.035 * s
        for i in range(cols):
            for j in range(rows):
                out.append(("square", (i + 1) / (cols + 1) * width,
                            (j + 1) / (rows + 1) * height, half))
        inset, t = 0.07 * s, 0.035 * s
        for cx, cy in ((inset, inset), (width - inset, inset),
                       (width - inset, height - inset), (inset, height - inset)):
            out.append(("tri", cx, cy + t / 3.0, t))
    elif pattern != "none":
        raise ValueError(f"unknown tracker pattern {pattern!r}")
    return out


def shapes(pattern, width, height):
    """The tracker's markers as geometry: `[(layer, kind, points)]`, drawn in
    order. `layer` is "rim" (the white outline) or "ink"; `kind` is "rect" with
    `(x0, y0, x1, y1)` or "tri" with `(cx, cy, t)`, an upward triangle of
    half-width `t` whose apex is `t` above its centre.

    Geometry rather than pixels, because the frontend draws the same tracker for
    its preview (`web/creator/screens.js`) and a list of shapes is what two
    rasterisers can be held to agree about.
    """
    s = min(width, height)
    out = []
    if pattern == "plus":
        inset = 0.09 * s
        arm, stroke = 0.045 * s, max(4.0, 0.012 * s)
        corners = ((inset, inset), (width - inset, inset),
                   (width - inset, height - inset), (inset, height - inset))
        outline = 0.012 * s
        centre = (width / 2, height / 2)
        for cx, cy in corners:
            out += [("rim", "rect", bar) for bar in _plus(cx, cy, arm + 4, stroke + 8)]
        out += [("rim", "rect", bar) for bar in _plus(*centre, 0.14 * s + outline,
                                                      0.06 * s + 2 * outline)]
        for cx, cy in corners:
            out += [("ink", "rect", bar) for bar in _plus(cx, cy, arm, stroke)]
        out += [("ink", "rect", bar) for bar in _plus(*centre, 0.14 * s, 0.06 * s)]
    elif pattern == "grid":
        cols, rows = grid_size(width, height)
        half = 0.035 * s
        for i in range(cols):
            for j in range(rows):
                cx, cy = (i + 1) / (cols + 1) * width, (j + 1) / (rows + 1) * height
                out.append(("ink", "rect", (cx - half, cy - half, cx + half, cy + half)))
        inset, t = 0.07 * s, 0.035 * s
        for cx, cy in ((inset, inset), (width - inset, inset),
                       (width - inset, height - inset), (inset, height - inset)):
            out.append(("ink", "tri", (cx, cy, t)))
    elif pattern != "none":
        raise ValueError(f"unknown tracker pattern {pattern!r}")
    return out


def _rect(mask, x0, y0, x1, y1):
    h, w = mask.shape
    x0, y0 = max(0, int(round(x0))), max(0, int(round(y0)))
    x1, y1 = min(w, int(round(x1))), min(h, int(round(y1)))
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True


def _triangle(mask, cx, cy, t):
    h, w = mask.shape
    y0, y1 = max(0, int(cy - t)), min(h, int(cy + t) + 1)
    x0, x1 = max(0, int(cx - t)), min(w, int(cx + t) + 1)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    ys = ys + 0.5
    xs = xs + 0.5
    # Width grows linearly from the apex (cy - t) to the base (cy + t).
    half = (ys - (cy - t)) / 2.0
    inside = (ys >= cy - t) & (ys <= cy + t) & (np.abs(xs - cx) <= half)
    mask[y0:y1, x0:x1] |= inside


def markers(pattern, width, height):
    """-> (ink, rim) boolean masks, shape (height, width).

    `rim` is the white outline the plus pattern draws around its ink so the
    markers read on any colour; the grid pattern has none.
    """
    layers = {"ink": np.zeros((height, width), dtype=bool),
              "rim": np.zeros((height, width), dtype=bool)}
    for layer, kind, points in shapes(pattern, width, height):
        (_rect if kind == "rect" else _triangle)(layers[layer], *points)
    return layers["ink"], layers["rim"] & ~layers["ink"]


def draw(pattern, colour, width, height):
    """-> the tracker, (height, width, 3) uint8."""
    fill = spec.COLOUR[colour].fill
    ink, rim = markers(pattern, width, height)
    out = np.empty((height, width, 3), dtype=np.uint8)
    out[:] = fill
    out[rim] = RIM
    out[ink] = INK
    return out


def holes(pattern, colour, width, height):
    """Where a keyed tracker has holes: the ink, and the rim wherever the rim is
    not itself the key colour. What a rectified frame's holes are compared to."""
    ink, rim = markers(pattern, width, height)
    return ink | rim if spec.COLOUR[colour].hue is not None else ink


# The trackers this pack drew into input/ before they moved to temp, by name.
_OLD_NAME = re.compile(r"^v\d+-(grid|plus|none)-(white|magenta|green|blue)-\d+x\d+\.png$")
_swept = False


def _sweep_input():
    """Take the trackers out of input/ that earlier versions left there.

    Once per process, and only files named the way this pack named them, in
    the folder only this pack wrote: nothing the user put in input/ is touched,
    and the folder goes only once it is empty.
    """
    global _swept
    if _swept:
        return
    _swept = True
    import folder_paths

    folder = os.path.join(folder_paths.get_input_directory(), spec.TRACKER_DIR)
    if not os.path.isdir(folder):
        return
    for name in os.listdir(folder):
        if _OLD_NAME.match(name):
            try:
                os.remove(os.path.join(folder, name))
            except OSError:
                pass
    try:
        os.rmdir(folder)
    except OSError:
        pass


def ensure(name):
    """The tracker named `name` (`spec.tracker_file`) on disk in ComfyUI's temp
    folder. -> its absolute path. Drawn from the name alone, and only once."""
    import folder_paths
    from PIL import Image

    relative = spec.tracker_path(name)
    parsed = spec.parse_tracker_file(os.path.basename(relative))
    if parsed is None or os.path.dirname(relative) != spec.TRACKER_DIR:
        raise ValueError(f"{name!r} is not a tracker this pack draws")
    _sweep_input()
    path = os.path.join(folder_paths.get_temp_directory(), relative)
    if os.path.isfile(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Written beside and renamed over, so a render that reads it mid-write
    # never loads half a picture.
    partial = f"{path}.{os.getpid()}.partial"
    Image.fromarray(draw(*parsed), "RGB").save(partial, format="PNG", compress_level=6)
    os.replace(partial, path)
    return path


def ensure_all(screens):
    """Every tracker `screens` (a tuple of `spec.Screen`) needs, on disk."""
    for screen in screens:
        ensure(screen.tracker_file)
