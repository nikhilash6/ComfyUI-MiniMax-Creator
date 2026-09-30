"""Putting the content on the screen: warp, matte, shading, grain, despill.

Everything here works on one frame at the reel's own size, float 0..1, and on a
crop around the screen wherever the work is local — a phone is a tenth of the
frame, and filtering the other nine tenths is time spent on pixels that come
back unchanged.

**The matte is the screen's shape, not the key's.** The prototype built it from
the keyed pixels, and the pixels along the screen's edge that H3 blended half
into the bezel fell under the key's thresholds, stayed outside the matte, and
showed as the faint coloured rim. Here the keyed region is grown by a pixel and
clipped to the fitted quad grown by a pixel and a half: the rim is inside the
matte and is covered, the rounded corners still come from the key, and nothing
outside the screen's own four sides is touched.

**Holes in the key are kept or covered by what they are.** A hole where the
tracker has a marker is a marker, and is covered. A wide pill across the top of
the screen is the dynamic island H3 draws, and stays on top, as does anything
else the tracker does not explain — a finger, a pen, a reflection of something
in front. A thin bar along the bottom is the home indicator, and is covered.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

from . import spec, track, tracker

# Everything below was tuned on 768x1024 frames and is scaled by the frame's
# short side over this.
REFERENCE_SHORT = 768.0

GROW_KEY_PX = 1.0
GROW_QUAD_PX = 1.5
# H3 draws the markers a little softer and larger than the tracker has them.
GROW_MARKER_PX = 3.0
# How much larger than its drawn shape a marker's blob may be and still be the
# marker: room for H3 drawing it a little large, and for a corner marker that
# runs into the bezel of a rounded corner. Anything bigger has something in
# front of it.
MARKER_BLOB = 4.0
GLASS_INSET_PX = 1.5
# How far inside the screen's edge the rim is still closed.
EDGE_BAND_PX = 3.0
FEATHER_PX = 0.7
SHADE_SIGMA = 8.0
SHADE_CLAMP = (0.85, 1.12)
GLOW_GAIN = 0.96
GLOW_SIGMA = 3.0
GLOW_MIX = 0.08
GRAIN = 2.2 / 255.0
DESPILL_RING_PX = 15.0
# The dynamic island: a pill at least this many times wider than tall, in the
# screen's top eighth. The home bar: at least this many times wider, in its
# bottom tenth.
ISLAND_ASPECT = 2.5
ISLAND_BAND = 0.08
HOME_ASPECT = 4.0
HOME_BAND = 0.10
# A hole counts as a marker when this share of it lies on the tracker's markers.
MARKER_SHARE = 0.5


def scale_of(frame):
    return min(frame.shape[0], frame.shape[1]) / REFERENCE_SHORT


# ---- the content -----------------------------------------------------------------


def fitted(picture, width, height, fit):
    """A content picture (H, W, 3 uint8) -> (height, width, 3) float, fitted by
    `fit`: `fill` crops to cover, `fit` letterboxes in black, `stretch` stretches.
    """
    source = Image.fromarray(picture)
    sw, sh = source.size
    if fit == "stretch":
        out = source.resize((width, height), Image.LANCZOS)
    else:
        scale = max(width / sw, height / sh) if fit == "fill" else min(width / sw, height / sh)
        rw, rh = max(1, round(sw * scale)), max(1, round(sh * scale))
        resized = source.resize((rw, rh), Image.LANCZOS)
        out = Image.new("RGB", (width, height))
        out.paste(resized, ((width - rw) // 2, (height - rh) // 2))
    return np.asarray(out, dtype=np.float32) / 255.0


class Content:
    """One screen's content as a pyramid, so a screen shown small samples a level
    near its own size rather than aliasing a large one down."""

    def __init__(self, picture, width, height, fit):
        base = fitted(picture, width, height, fit)
        self.levels = [torch.from_numpy(base).permute(2, 0, 1).unsqueeze(0)]
        while min(self.levels[-1].shape[-2:]) > 64 and len(self.levels) < 6:
            self.levels.append(F.avg_pool2d(self.levels[-1], 2))
        self.width, self.height = width, height

    def level_for(self, corners):
        """The pyramid level whose size is nearest the screen's on the frame."""
        span = max(np.linalg.norm(corners[1] - corners[0]),
                   np.linalg.norm(corners[2] - corners[3]))
        ratio = self.width / max(span, 1.0)
        return int(np.clip(np.floor(np.log2(max(ratio, 1.0))), 0, len(self.levels) - 1))


def warp(content, corners, box):
    """The content drawn onto `corners` (tracker order) over the frame crop `box`
    (y0, y1, x0, x1). -> (h, w, 3) float for the crop."""
    y0, y1, x0, x1 = box
    level = content.level_for(corners)
    picture = content.levels[level]
    lh, lw = picture.shape[-2:]
    # Frame -> content, in the chosen level's pixels.
    h = track.homography(corners, track.canonical_corners(lw, lh))
    ys, xs = np.mgrid[y0:y1, x0:x1]
    xy = track.apply(h, np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], axis=1))
    grid = np.stack([xy[:, 0] / lw * 2.0 - 1.0, xy[:, 1] / lh * 2.0 - 1.0], axis=1)
    grid = torch.from_numpy(grid.reshape(1, y1 - y0, x1 - x0, 2).astype(np.float32))
    out = F.grid_sample(picture, grid, mode="bilinear", padding_mode="border",
                        align_corners=False)
    return out[0].permute(1, 2, 0).numpy()


# ---- the matte -------------------------------------------------------------------


def polygon(corners, shape, grow=0.0):
    """A quad rasterised with 4x4 supersampling -> coverage (h, w) float, grown
    outward by `grow` pixels."""
    centre = corners.mean(axis=0)
    if grow:
        # Each side pushed out along its own normal, the corners re-intersected.
        lines = []
        for side in range(4):
            a, b = corners[side], corners[(side + 1) % 4]
            d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
            n = np.array([-d[1], d[0]])
            if np.dot(a - centre, n) < 0:
                n = -n
            lines.append((a + grow * n, d))
        corners = np.array([track._intersect(lines[k - 1], lines[k]) for k in range(4)])
    h, w = shape
    sub = 4
    ys, xs = np.mgrid[0:h * sub, 0:w * sub]
    px = (xs + 0.5) / sub
    py = (ys + 0.5) / sub
    inside = np.ones_like(px, dtype=bool)
    sign = None
    for side in range(4):
        a, b = corners[side], corners[(side + 1) % 4]
        cross = (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])
        if sign is None:
            sign = np.sign((b[0] - a[0]) * (centre[1] - a[1]) - (b[1] - a[1]) * (centre[0] - a[0]))
        inside &= cross * sign >= 0
    return inside.reshape(h, sub, w, sub).mean(axis=(1, 3))


def _disk(radius):
    r = max(1, int(np.ceil(radius)))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= radius * radius + 0.25


def classify_holes(filled, keyed, corners, screen, answer_holes):
    """-> (kept, covered) boolean masks of the key's holes.

    `answer_holes` is the tracker's own marker mask at full tracker size;
    `corners` are in tracker order, so tracker y is the screen's own up.
    """
    holes = filled & ~keyed
    labels, count = ndimage.label(holes)
    kept = np.zeros_like(holes)
    covered = np.zeros_like(holes)
    if not count:
        return kept, covered
    tw, th = screen.width, screen.height
    to_tracker = track.homography(corners, track.canonical_corners(tw, th))
    for index, where in enumerate(ndimage.find_objects(labels), start=1):
        blob = labels[where] == index
        ys, xs = np.nonzero(blob)
        ys = ys + where[0].start + 0.5
        xs = xs + where[1].start + 0.5
        uv = track.apply(to_tracker, np.stack([xs, ys], axis=1))
        u = np.clip(uv[:, 0].astype(int), 0, tw - 1)
        v = np.clip(uv[:, 1].astype(int), 0, th - 1)
        on_marker = answer_holes[v, u].mean() if len(u) else 0.0
        width = uv[:, 0].max() - uv[:, 0].min() + 1
        height = uv[:, 1].max() - uv[:, 1].min() + 1
        lowest = uv[:, 1].max() / th
        highest = uv[:, 1].min() / th
        target = labels == index
        if on_marker >= MARKER_SHARE:
            covered |= target
        elif lowest <= ISLAND_BAND and width >= ISLAND_ASPECT * height:
            kept |= target
        elif highest >= 1.0 - HOME_BAND and width >= HOME_ASPECT * height:
            covered |= target
        else:
            kept |= target
    return kept, covered


def projected(answer, corners, shape):
    """The tracker's markers drawn where the fit puts them -> (h, w) bool."""
    th, tw = answer.shape
    h, w = shape
    to_tracker = track.homography(corners, track.canonical_corners(tw, th))
    ys, xs = np.mgrid[0:h, 0:w]
    uv = track.apply(to_tracker, np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], axis=1))
    u = np.floor(uv[:, 0]).astype(int)
    v = np.floor(uv[:, 1]).astype(int)
    inside = (u >= 0) & (u < tw) & (v >= 0) & (v < th)
    out = np.zeros(h * w, dtype=bool)
    out[inside] = answer[v[inside], u[inside]]
    return out.reshape(h, w)


def visible_markers(markers, region, inside, reach):
    """The marker blobs the frame actually shows, to be covered.

    Read off the frame rather than off the projection: H3 draws the markers
    near, not exactly on, where the tracker has them, and a test against the
    projected shape reads every misregistered marker as covered by something.
    So each projected marker picks out the real non-key blob it lands on. A
    blob about a marker's size is the marker — a corner one that runs into a
    rounded corner's bezel is still only a little larger. A blob many times
    that is the marker merged into something standing in front of it, and is
    left alone: covering it would paint the content onto whoever that is.

    Only the blob's pixels within `reach` of the drawn marker are taken: a
    corner marker's blob includes the rounded corner's bezel wedge, which is
    the phone's and not the tracker's.
    """
    loose = inside & ~region
    labels, _ = ndimage.label(loose)
    sizes = np.bincount(labels.ravel())
    marks, count = ndimage.label(markers)
    out = np.zeros_like(markers)
    for index, where in enumerate(ndimage.find_objects(marks), start=1):
        blob = marks[where] == index
        hit = np.unique(labels[where][blob])
        near = None
        for part in hit[hit > 0]:
            if sizes[part] <= MARKER_BLOB * blob.sum():
                if near is None:
                    near = ndimage.binary_dilation(marks == index, structure=_disk(reach))
                out |= (labels == part) & near
    return out


def matte(filled, markers, corners_image, kept, scale):
    """Coverage of the new content over the crop, 0..1.

    `filled` is the screen's key with its enclosed holes filled; `markers` the
    visible markers; `kept` the holes that stay on top. The growth that closes
    the rim is only along the screen's own edge: grown everywhere, it would
    paint a pixel of content over every arm and finger in front of the screen.
    """
    grow = GROW_KEY_PX * max(scale, 1.0)
    base = (filled | markers) & ~kept
    quad = polygon(corners_image, filled.shape, grow=GROW_QUAD_PX * max(scale, 1.0))
    core = polygon(corners_image, filled.shape, grow=-EDGE_BAND_PX * max(scale, 1.0)) > 0.5
    rim = ndimage.binary_dilation(base, structure=_disk(grow)) & ~core
    alpha = np.minimum((base | rim).astype(np.float32), quad)
    # The kept holes are cut back out after the growth, so the island keeps its
    # own edge rather than losing a pixel of it to the dilation.
    alpha[kept] = 0.0
    return ndimage.gaussian_filter(alpha, FEATHER_PX * scale)


@dataclass
class Layers:
    """What a screen's matte was made of, for the composite and the debug view."""
    box: tuple
    alpha: np.ndarray
    keyed: np.ndarray
    kept: np.ndarray
    covered: np.ndarray
    markers: np.ndarray
    inside: np.ndarray


def layers(frame, screen, corners, answer_holes):
    """The matte for one screen over the crop around it, or None off-frame."""
    scale = scale_of(frame)
    pad = DESPILL_RING_PX * scale + 4
    box = box_around(corners, frame.shape[:2], pad)
    y0, y1, x0, x1 = box
    if y1 - y0 < 4 or x1 - x0 < 4:
        return None
    crop = frame[y0:y1, x0:x1]
    local = corners - np.array([x0, y0])
    keyed = track.key(crop, screen.colour)
    inside = polygon(image_order(local), keyed.shape, grow=GROW_QUAD_PX * scale + 2) > 0.5
    # Every keyed piece under the quad, not the largest: a person in front of
    # the screen can cut it in two, and both halves are screen.
    region = keyed & inside
    filled = ndimage.binary_fill_holes(region)
    kept, covered = classify_holes(filled, region, local, screen, answer_holes)
    # Searched for inside the glass, not the grown quad: the grown quad takes in
    # a strip of bezel all the way round, which would join every corner marker
    # to every other and to the bezel itself.
    glass = polygon(image_order(local), keyed.shape, grow=-GLASS_INSET_PX * max(scale, 1.0)) > 0.5
    markers = visible_markers(projected(answer_holes, local, keyed.shape), region, glass,
                              GROW_MARKER_PX * max(scale, 1.0))
    # A pixel out for the soft edge H3 draws round each marker.
    markers = ndimage.binary_dilation(markers, structure=_disk(max(scale, 1.0))) & inside & ~kept
    alpha = matte(filled, markers, image_order(local), kept, scale)
    return Layers(box, alpha, region, kept, covered, markers, inside)


# ---- the look ----------------------------------------------------------------------


def shading(frame, keyed, markers_covered, scale):
    """H3's own light across the screen: the key's brightness, marker holes filled
    in from around them, relative to its median and held to a narrow band."""
    value = frame.max(axis=-1)
    use = keyed & ~markers_covered
    if not use.any():
        return np.ones(value.shape, dtype=np.float32)
    sigma = SHADE_SIGMA * scale
    weight = ndimage.gaussian_filter(use.astype(np.float32), sigma)
    blurred = ndimage.gaussian_filter(np.where(use, value, 0.0).astype(np.float32), sigma)
    field = blurred / np.maximum(weight, 1e-4)
    median = float(np.median(value[use]))
    return np.clip(field / max(median, 1e-4), *SHADE_CLAMP).astype(np.float32)


def glow_and_grain(content, scale, rng):
    blurred = np.stack([ndimage.gaussian_filter(content[..., c], GLOW_SIGMA * scale)
                        for c in range(3)], axis=-1)
    out = content * GLOW_GAIN + blurred * GLOW_MIX
    return out + rng.normal(0.0, GRAIN, size=out.shape).astype(np.float32)


def despill(frame, zone, colour):
    """The key colour's light taken back out of `zone` (float 0..1 weights)."""
    hue = spec.COLOUR[colour].hue
    if hue is None:
        # A neutral key casts neutral light, which is what a lit screen casts.
        return frame
    r, g, b = frame[..., 0], frame[..., 1], frame[..., 2]
    out = frame.copy()
    if colour == "magenta":
        excess = np.maximum(0.0, np.minimum(r, b) - g) * zone
        out[..., 0] -= 0.8 * excess
        out[..., 2] -= 0.8 * excess
        out[..., 1] += 0.35 * excess
    else:
        channel = {"green": 1, "blue": 2}[colour]
        others = [c for c in range(3) if c != channel]
        excess = np.maximum(0.0, frame[..., channel]
                            - np.maximum(frame[..., others[0]], frame[..., others[1]])) * zone
        out[..., channel] -= 0.8 * excess
        for c in others:
            out[..., c] += 0.35 * excess
    return out


# ---- one frame -------------------------------------------------------------------------


def box_around(corners, shape, pad):
    h, w = shape
    x0 = int(max(0, np.floor(corners[:, 0].min() - pad)))
    x1 = int(min(w, np.ceil(corners[:, 0].max() + pad)))
    y0 = int(max(0, np.floor(corners[:, 1].min() - pad)))
    y1 = int(min(h, np.ceil(corners[:, 1].max() + pad)))
    return y0, y1, x0, x1


def image_order(corners):
    return track._order(np.asarray(corners))


def compose(frame, screen, corners, picture, answer_holes, rng, drawn=None):
    """One screen onto one frame. `frame` float (H, W, 3) is changed in place and
    returned; `corners` are the smoothed, tracker-ordered corners; `picture` is
    the `Content` for this frame. `drawn`, a list, is handed the `Layers`."""
    made = layers(frame, screen, corners, answer_holes)
    if made is None:
        return frame
    if drawn is not None:
        drawn.append(made)
    scale = scale_of(frame)
    y0, y1, x0, x1 = made.box
    crop = frame[y0:y1, x0:x1]
    alpha = made.alpha

    content = warp(picture, corners, made.box)
    content = content * shading(crop, made.keyed, made.covered | made.markers, scale)[..., None]
    content = glow_and_grain(content, scale, rng)

    ring = ndimage.binary_dilation(alpha > 0.5, structure=_disk(DESPILL_RING_PX * scale)) & (alpha < 0.5)
    around_kept = ndimage.binary_dilation(made.kept, structure=_disk(4 * scale)) & (alpha < 0.5)
    zone = ndimage.gaussian_filter((ring | around_kept).astype(np.float32), 2.0 * scale)
    plate = despill(crop, zone, screen.colour)

    a = alpha[..., None]
    frame[y0:y1, x0:x1] = np.clip(plate * (1.0 - a) + content * a, 0.0, 1.0)
    return frame


# ---- the debug view ---------------------------------------------------------------------

DEBUG_TINT = {"content": (0.2, 0.9, 0.35), "occluder": (1.0, 0.25, 0.2),
              "solved": (1.0, 0.85, 0.1), "filled": (1.0, 0.45, 0.0), "marker": (0.2, 0.85, 1.0)}


def _line(image, a, b, colour):
    steps = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) * 2 + 1
    for t in np.linspace(0.0, 1.0, steps):
        x, y = a + (b - a) * t
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < image.shape[0] and 0 <= xi < image.shape[1]:
            image[max(0, yi - 1):yi + 1, max(0, xi - 1):xi + 1] = colour


def overlay(raw, screen, corners, solved, drawn):
    """What the pass saw on one frame, drawn over the raw render: the new
    content's matte in green, what stayed on top of the screen in red, the
    quad in yellow where it was solved and orange where it was filled in from
    neighbouring frames, and the tracker's markers where the quad puts them."""
    image = raw * 0.55
    if corners is None:
        return image
    if drawn is not None:
        y0, y1, x0, x1 = drawn.box
        crop = image[y0:y1, x0:x1]
        green = np.array(DEBUG_TINT["content"])
        red = np.array(DEBUG_TINT["occluder"])
        a = drawn.alpha[..., None]
        crop[:] = crop * (1 - 0.5 * a) + green * 0.5 * a
        held = (drawn.inside & (drawn.alpha < 0.5))[..., None]
        crop[:] = np.where(held, crop * 0.5 + red * 0.5, crop)
    colour = DEBUG_TINT["solved" if solved else "filled"]
    ordered = image_order(corners)
    for k in range(4):
        _line(image, ordered[k], ordered[(k + 1) % 4], colour)
    anchors = np.array([(x, y) for _, x, y, _ in
                        tracker.anchors(screen.pattern, screen.width, screen.height)]).reshape(-1, 2)
    if not len(anchors):
        # A plain screen has no markers to show.
        return image
    plane = track.fit_homography(track.canonical_corners(screen.width, screen.height), corners)
    for x, y in track.apply(plane, anchors):
        xi, yi = int(round(x)), int(round(y))
        if 1 <= yi < image.shape[0] - 1 and 1 <= xi < image.shape[1] - 1:
            image[yi - 1:yi + 2, xi - 1:xi + 2] = DEBUG_TINT["marker"]
    return image


def answer_holes(screen):
    return tracker.holes(screen.pattern, screen.colour, screen.width, screen.height)
