"""Finding the screen in a frame, and following it through a shot.

The prototype used OpenCV (`minAreaRect`, `fitLine`, `connectedComponents`).
This pack has no dependencies and OpenCV is not one of core's, so the same steps
are written against numpy and `scipy.ndimage`, which are — the rule
`control.py` already follows.

**Where the screen is comes from its markers, not its outline.** That is how a
screen insert is tracked in production — a planar track of the surface, with
whatever passes in front of it held out — and the reason is the case the first
version of this module got wrong: a person walking in front of a TV. The key's
outline is the screen's outline only while nothing covers it; once an arm
crosses an edge, the outline follows the arm, and a fit to it follows the arm
too. The markers are points on the screen's own plane. A covered marker is
simply missing, and the ones that are left still say where the plane is.

So each frame is *observed* once (`observe`): the key, the marker-shaped holes
in it, and — where the screen is whole enough — a four-sided fit to its outline,
which only ever serves as a starting point. Then each shot is *followed*
(`follow`): from its clearest frame, forwards and backwards, each frame's holes
are matched to the tracker's markers where the frames before predict them, and
the plane is solved from the matches that agree (RANSAC over a homography).
A frame with too few markers left is filled in from the frames either side.

**Which corner is which** is read off the markers too: the starting fit is tried
under each quarter turn and the turn whose holes line up with the tracker's
markers is the one the device is held at — so a phone on its side shows its
content on its side, the way a phone does. The plus pattern is symmetric under a
half turn, and there the upright reading of the two wins.

**A cut** is a change of picture (the neural pass's own luma test), or the
markers being lost where the prediction put them and found again somewhere
else. A corner moving fast is not one: a tracked plane may move as fast as the
camera does.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from . import spec, tracker

# ---- the key --------------------------------------------------------------------
#
# The prototype's magenta thresholds in OpenCV units were H 135-170, S >= 80,
# V >= 120; as degrees and fractions that is a band either side of 305°, S >=
# 0.31 and V >= 0.47. The same widths serve every hue here, and V goes a little
# lower because a screen turned away from the light falls off fast.
HUE_BAND = 35.0
MIN_SAT = 0.31
MIN_VAL = 0.35
# A neutral key is bright and colourless, and bright is relative: white on the
# lab came back around V 0.87, S 0.08, beside a grey wall at V 0.66 that a fixed
# floor low enough for a dim room would take in too. So the floor is a share of
# the brightest colourless pixels in the frame (the screen, where there is one),
# and never under the absolute one. A white wall brighter still clears it as
# well, which is why a component is then chosen by its markers.
NEUTRAL_MAX_SAT = 0.2
NEUTRAL_MIN_VAL = 0.6
NEUTRAL_RELATIVE = 0.85
NEUTRAL_PEAK = 99.5
# **The key grows from its core.** A screen is not evenly lit — a TV in a dark
# room falls off towards one side — and a threshold that takes the bright side
# cleanly drops the dim one, which then stays on screen as a strip of tracker.
# So the thresholds above only find the core; the key is every pixel that
# clears these looser ones *and is connected to a core*, which reaches the whole
# screen without reaching a grey wall across the room.
WEAK_HUE_BAND = 45.0
WEAK_SAT = 0.18
WEAK_VAL = 0.2
WEAK_NEUTRAL_SAT = 0.28
WEAK_NEUTRAL_RELATIVE = 0.55
WEAK_NEUTRAL_MIN = 0.3

# ---- the starting fit ------------------------------------------------------------
SIDE_MIDDLE = 0.75      # the share of each side the line is fitted to
HUBER_PX = 1.0
MIN_SIDE_POINTS = 8
# A screen smaller than this share of the frame is too few pixels to fit four
# sides to, and is reported as not found.
MIN_AREA = 0.002
# The quad has to account for the region it was fitted to, to within this.
MAX_AREA_MISMATCH = 0.25
# The marker check: the least overlap between the frame's holes and the
# tracker's markers for a fit to count as the tracker at all — and the overlap a
# fit needs before a shot is started from it, which is higher: a shot is only
# ever started from a frame where the screen is plainly all there.
MIN_MARKER_SCORE = 0.12
SEED_SCORE = 0.3
# The resolution the marker check samples the tracker at, on its long side.
CHECK_SIZE = 96

# ---- the edges ----------------------------------------------------------------------
# A side is moved onto the key's outline when outline points lie within this
# share of its length of it (or three pixels), along at least this share of its
# middle, on a line straighter than this many pixels from it at the median. An
# arm across the edge fails all three.
SIDE_BAND = 0.03
SIDE_SUPPORT = 0.5
SIDE_STRAIGHT = 0.8
# A screen with no markers is followed by its edges alone: a frame needs this
# many of the four to be solved, and an edge only counts where it has turned
# less than this many degrees from where it was looked for. On the lab, an edge
# half behind a passing person fitted a line a degree or two off true, and a
# plain screen has nothing else to check it against.
PLAIN_SIDES = 3
PLAIN_TURN = 1.0
# With markers, a side's edge is only put back where at least this share of the
# shot's solved frames saw it.
EDGE_VOTES = 0.3

# ---- following the markers --------------------------------------------------------
# A hole is matched to the marker predicted nearest it, within this share of the
# distance between neighbouring markers, and only if it is about the right size.
MATCH_RADIUS = 0.45
MATCH_AREA = (0.12, 5.0)
# The plane needs this many markers that agree; fewer, and the frame is filled
# in from its neighbours.
MIN_MATCHES = 4
RANSAC_ROUNDS = 96
# How far a marker may sit from where the solved plane puts it, in pixels, and
# as a share of the marker spacing when that is larger.
INLIER_PX = 1.5
INLIER_SHARE = 0.08
# How many frames in a row may go unsolved before the prediction stops moving
# and just holds where the screen last was.
MAX_COAST = 6

# ---- the shot -----------------------------------------------------------------------
# How far the screen has to have moved when its markers are lost and found again
# for that to be a cut rather than a fit that disagrees with the prediction by a
# little: this many pixels on a 1280-pixel diagonal, or this share of the
# screen's own diagonal, whichever is more. A cut H3 makes is caught by the
# change of picture first; this is for a cut that keeps the room and moves the
# screen, which moves it a long way.
CUT_PX = 10.0
CUT_SHARE = 0.25
# The mean absolute luma difference, 0..1, that the neural pass also reads as a
# cut (`mlxdlss/temporal.py`).
CUT_LUMA = 0.3
SMOOTH_SIGMA = 1.2
SMOOTH_RADIUS = 3


def hsv(frame):
    """(H, W, 3) float 0..1 -> hue in degrees, saturation, value."""
    r, g, b = frame[..., 0], frame[..., 1], frame[..., 2]
    top = frame.max(axis=-1)
    low = frame.min(axis=-1)
    span = top - low
    sat = np.where(top > 1e-6, span / np.maximum(top, 1e-6), 0.0)
    safe = np.maximum(span, 1e-6)
    hue = np.where(top == r, (g - b) / safe % 6.0,
                   np.where(top == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0)) * 60.0
    return np.where(span > 1e-6, hue, 0.0), sat, top


def _grown(core, loose):
    """The loose mask's pieces that hold some of the core."""
    if not core.any():
        return core
    labels, _ = ndimage.label(loose | core)
    held = np.unique(labels[core])
    return np.isin(labels, held[held > 0])


def key(frame, colour):
    """Where the frame shows the key colour. `frame` is float 0..1."""
    hue, sat, val = hsv(frame)
    target = spec.COLOUR[colour].hue
    if target is None:
        grey = sat <= NEUTRAL_MAX_SAT
        if not grey.any():
            return grey
        peak = float(np.percentile(val[grey], NEUTRAL_PEAK))
        core = grey & (val >= max(NEUTRAL_MIN_VAL, NEUTRAL_RELATIVE * peak))
        loose = (sat <= WEAK_NEUTRAL_SAT) & (val >= max(WEAK_NEUTRAL_MIN, WEAK_NEUTRAL_RELATIVE * peak))
        return _grown(core, loose)
    distance = np.abs((hue - target + 180.0) % 360.0 - 180.0)
    core = (distance <= HUE_BAND) & (sat >= MIN_SAT) & (val >= MIN_VAL)
    loose = (distance <= WEAK_HUE_BAND) & (sat >= WEAK_SAT) & (val >= WEAK_VAL)
    return _grown(core, loose)


@dataclass
class Found:
    corners: np.ndarray        # (4, 2) x, y in image order: TL, TR, BR, BL
    turns: np.ndarray          # (4,) marker score under each quarter turn
    area: float


def _hole_count(component, filled):
    labels, count = ndimage.label(filled & ~component)
    if not count:
        return 0
    sizes = np.bincount(labels.ravel())[1:]
    return int((sizes >= 4).sum())


def _choose(mask, previous=None):
    """The keyed component most like a tracker -> (filled region, component), or
    (None, None). Size counts, and marker-shaped holes count for more: a white
    wall is big and has none."""
    labels, count = ndimage.label(mask)
    if not count:
        return None, None
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    floor = MIN_AREA * mask.size
    best, best_score = None, 0.0
    for index in np.argsort(sizes)[::-1][:4]:
        if sizes[index] < floor:
            break
        component = labels == index
        filled = ndimage.binary_fill_holes(component)
        holes = _hole_count(component, filled)
        score = float(sizes[index]) * (1.0 + min(holes, 13))
        if previous is not None:
            # A screen stays where it was: the component under the last fit's
            # centre wins a tie with a brighter patch elsewhere.
            cx, cy = previous.mean(axis=0)
            iy, ix = int(round(cy)), int(round(cx))
            if 0 <= iy < filled.shape[0] and 0 <= ix < filled.shape[1] and filled[iy, ix]:
                score *= 2.0
        if score > best_score:
            best, best_score = (filled, component), score
    return best if best else (None, None)


def _outline(filled):
    edge = filled & ~ndimage.binary_erosion(filled)
    ys, xs = np.nonzero(edge)
    return np.stack([xs + 0.5, ys + 0.5], axis=1).astype(np.float64)


def _order(corners):
    """Four points -> TL, TR, BR, BL (clockwise on screen, y down)."""
    centre = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - centre[1], corners[:, 0] - centre[0])
    ring = corners[np.argsort(angles)]
    start = int(np.argmin(ring.sum(axis=1)))
    return np.roll(ring, -start, axis=0)


def _initial(filled, points):
    """Rough corners: the outline's extremes along the region's diagonals."""
    ys, xs = np.nonzero(filled)
    pixels = np.stack([xs, ys], axis=1).astype(np.float64)
    centre = pixels.mean(axis=0)
    cov = np.cov((pixels - centre).T)
    _, vectors = np.linalg.eigh(cov)
    u, v = vectors[:, 1], vectors[:, 0]
    rel = points - centre
    pu, pv = rel @ u, rel @ v
    # Normalised, so a long screen's diagonals are its corners and not the
    # middles of its long sides.
    pu = pu / max(np.abs(pu).max(), 1e-6)
    pv = pv / max(np.abs(pv).max(), 1e-6)
    corners = np.array([points[np.argmax(su * pu + sv * pv)]
                        for su, sv in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
    return _order(corners)


def _fit_line(points):
    """Huber-weighted orthogonal line fit -> (point on line, unit direction)."""
    weights = np.ones(len(points))
    for _ in range(6):
        centre = (points * weights[:, None]).sum(axis=0) / weights.sum()
        rel = points - centre
        cov = (rel * weights[:, None]).T @ rel
        _, vectors = np.linalg.eigh(cov)
        direction = vectors[:, 1]
        normal = np.array([-direction[1], direction[0]])
        residual = np.abs(rel @ normal)
        weights = np.where(residual <= HUBER_PX, 1.0, HUBER_PX / np.maximum(residual, 1e-9))
    return centre, direction


def _intersect(a, b):
    (p, d), (q, e) = a, b
    matrix = np.array([d, -e]).T
    if abs(np.linalg.det(matrix)) < 1e-9:
        return None
    t = np.linalg.solve(matrix, q - p)
    return p + t[0] * d


def _polygon_area(corners):
    x, y = corners[:, 0], corners[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def fit_quad(filled):
    """A filled screen region -> (4, 2) corners TL, TR, BR, BL, or None."""
    points = _outline(filled)
    if len(points) < 4 * MIN_SIDE_POINTS:
        return None
    corners = _initial(filled, points)
    centre = filled.nonzero()
    centre = np.array([centre[1].mean() + 0.5, centre[0].mean() + 0.5])
    margin = (1.0 - SIDE_MIDDLE) / 2.0
    for rounds in range(3):
        sides = []
        for side in range(4):
            a, b = corners[side], corners[(side + 1) % 4]
            length = np.linalg.norm(b - a)
            if length < 4:
                return None
            direction = (b - a) / length
            normal = np.array([-direction[1], direction[0]])
            rel = points - a
            sides.append((rel @ direction / length, np.abs(rel @ normal), length))
        # The first round's corners sit on the rounded corners' arcs, well
        # inside the true ones, so a band around their chords misses the sides
        # entirely. Each outline point goes to the chord it is nearest instead;
        # once the corners are intersections of fitted sides, the band holds.
        nearest = np.argmin(np.stack([across for _, across, _ in sides]), axis=0)
        lines = []
        for side, (along, across, length) in enumerate(sides):
            if rounds == 0:
                near = nearest == side
            else:
                near = across <= max(3.0, 0.03 * length)
            chosen = points[(along >= margin) & (along <= 1.0 - margin) & near]
            if len(chosen) < MIN_SIDE_POINTS:
                return None
            point, line_dir = _fit_line(chosen)
            # The outline is the centres of the region's edge pixels, half a
            # pixel inside the edge they sample. Moved out by that half pixel,
            # away from the region's centre.
            line_normal = np.array([-line_dir[1], line_dir[0]])
            if np.dot(point - centre, line_normal) < 0:
                line_normal = -line_normal
            lines.append((point + 0.5 * line_normal, line_dir))
        fitted = []
        for side in range(4):
            corner = _intersect(lines[side - 1], lines[side])
            if corner is None:
                return None
            fitted.append(corner)
        corners = _order(np.array(fitted))
    area = _polygon_area(corners)
    region = float(filled.sum())
    if region <= 0 or abs(area - region) / region > MAX_AREA_MISMATCH:
        return None
    return corners


def homography(src, dst):
    """The 3x3 map taking four `src` points onto four `dst` points."""
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    h = vt[-1].reshape(3, 3)
    return h / h[2, 2]


def apply(h, points):
    points = np.asarray(points, dtype=np.float64)
    ones = np.ones((len(points), 1))
    mapped = np.hstack([points, ones]) @ h.T
    return mapped[:, :2] / mapped[:, 2:3]


def canonical_corners(width, height):
    return np.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]])


def turned(corners, turn):
    """The quad's corners in tracker order when the tracker is turned by `turn`
    quarter turns: tracker corner k lands on image corner k + turn."""
    return np.roll(corners, -turn, axis=0)


class Answer:
    """What a tracker's markers look like, sampled small, for the marker check."""

    def __init__(self, pattern, colour, width, height):
        scale = CHECK_SIZE / max(width, height)
        self.width = max(8, int(round(width * scale)))
        self.height = max(8, int(round(height * scale)))
        self.tracker = (width, height)
        full = tracker.holes(pattern, colour, width, height)
        ys = ((np.arange(self.height) + 0.5) / self.height * height).astype(int)
        xs = ((np.arange(self.width) + 0.5) / self.width * width).astype(int)
        self.holes = full[np.ix_(ys, xs)]
        gy, gx = np.mgrid[0:self.height, 0:self.width]
        self.grid = np.stack([(gx.ravel() + 0.5) / self.width * width,
                              (gy.ravel() + 0.5) / self.height * height], axis=1)

    def scores(self, corners, holes):
        """The overlap (IoU) of the frame's holes with the markers, under each
        of the four quarter turns."""
        out = np.zeros(4)
        tw, th = self.tracker
        for turn in range(4):
            h = homography(canonical_corners(tw, th), turned(corners, turn))
            xy = apply(h, self.grid)
            xs = np.clip(np.round(xy[:, 0] - 0.5).astype(int), 0, holes.shape[1] - 1)
            ys = np.clip(np.round(xy[:, 1] - 0.5).astype(int), 0, holes.shape[0] - 1)
            seen = holes[ys, xs].reshape(self.height, self.width)
            union = (seen | self.holes).sum()
            out[turn] = (seen & self.holes).sum() / union if union else 0.0
        return out


def _normalised(points):
    centre = points.mean(axis=0)
    spread = np.sqrt(((points - centre) ** 2).sum(axis=1).mean()) or 1.0
    scale = np.sqrt(2.0) / spread
    t = np.array([[scale, 0, -scale * centre[0]], [0, scale, -scale * centre[1]], [0, 0, 1]])
    return apply(t, points), t


def fit_homography(src, dst):
    """The least-squares plane through any number (>= 4) of point pairs, with the
    points normalised first so the solve is well conditioned. -> 3x3, or None."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if len(src) < 4:
        return None
    a, ta = _normalised(src)
    b, tb = _normalised(dst)
    rows = []
    for (x, y), (u, v) in zip(a, b):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, sv, vt = np.linalg.svd(np.asarray(rows))
    if sv[-2] < 1e-9:
        return None
    h = np.linalg.inv(tb) @ vt[-1].reshape(3, 3) @ ta
    if abs(h[2, 2]) < 1e-12:
        return None
    return h / h[2, 2]


# ---- one frame ----------------------------------------------------------------------


@dataclass
class Seen:
    """What one frame shows of a screen: the marker-shaped holes in the key, as
    `(x, y, area)` rows; the key's outline, as points; and — where the screen is
    whole enough to fit — a starting fit. `luma` is for the cut test."""
    holes: np.ndarray
    outline: np.ndarray
    start: Found | None
    luma: np.ndarray


def _big(mask):
    """The key's pieces large enough to be a screen, holes filled."""
    labels, count = ndimage.label(mask)
    if not count:
        return mask
    sizes = np.bincount(labels.ravel())
    floor = MIN_AREA * mask.size * 0.25
    keep = np.nonzero(sizes >= floor)[0]
    keep = keep[keep > 0]
    return ndimage.binary_fill_holes(np.isin(labels, keep)) if len(keep) else mask


def _holes(mask, big):
    """Every hole the key closes around: its centre and its area."""
    enclosed = big & ~mask
    hole_labels, holes = ndimage.label(enclosed)
    if not holes:
        return np.zeros((0, 3))
    index = np.arange(1, holes + 1)
    areas = ndimage.sum(np.ones_like(enclosed, dtype=np.float32), hole_labels, index)
    centres = ndimage.center_of_mass(enclosed, hole_labels, index)
    rows = [(x + 0.5, y + 0.5, a) for (y, x), a in zip(centres, areas) if a >= 3]
    return np.array(rows, dtype=np.float64).reshape(-1, 3)


def observe(frame, screen, answer):
    """One frame (float 0..1) -> `Seen`."""
    mask = key(frame, screen.colour)
    start = None
    filled, component = _choose(mask)
    if filled is not None:
        corners = fit_quad(filled)
        if corners is not None:
            if screen.pattern == "none":
                # Nothing to check a fit against but its own shape. A screen
                # half hidden fits a quad too — the visible half, with the
                # person's edge for a side — so the score is how close the fit
                # comes to the screen's own proportions, under each turn.
                turns = np.array([_shape_score(turned(corners, t), screen.width / screen.height)
                                  for t in range(4)])
            else:
                turns = answer.scores(corners, filled & ~component)
            if turns.max() >= MIN_MARKER_SCORE:
                start = Found(corners=corners, turns=turns, area=float(filled.sum()))
    big = _big(mask)
    return Seen(holes=_holes(mask, big), outline=_outline(big).astype(np.float32),
                start=start, luma=luma_small(frame))


def _shape_score(corners, aspect):
    """1 for a quad in the tracker's proportions, falling off as it strays —
    allowing for the foreshortening a screen seen at an angle has."""
    wide = np.linalg.norm(corners[1] - corners[0]) + np.linalg.norm(corners[2] - corners[3])
    tall = np.linalg.norm(corners[3] - corners[0]) + np.linalg.norm(corners[2] - corners[1])
    return float(np.exp(-3.0 * abs(np.log(wide / max(tall, 1e-9)) - np.log(aspect))))


def locate(frame, screen, answer, previous=None):
    """One frame on its own -> its starting fit (`Found`), or None. What a
    still is read with, where there is no shot to follow."""
    return observe(frame, screen, answer).start


class Markers:
    """A tracker's markers as points to match: their centres, sizes and kinds."""

    def __init__(self, screen):
        rows = tracker.anchors(screen.pattern, screen.width, screen.height)
        self.points = np.array([(x, y) for _, x, y, _ in rows], dtype=np.float64).reshape(-1, 2)
        self.half = np.array([half for _, _, _, half in rows], dtype=np.float64)
        # How much of its bounding square each kind of marker fills, for the
        # size check: a triangle half of it, a plus about a third.
        fill = {"square": 1.0, "tri": 0.5, "cross": 0.25, "plus": 0.4}
        self.fill = np.array([fill[kind] for kind, _, _, _ in rows])
        self.width, self.height = screen.width, screen.height


def _local_scale(h, points):
    """How many image pixels one tracker pixel spans at each point."""
    dx = apply(h, points + [1.0, 0.0]) - apply(h, points)
    dy = apply(h, points + [0.0, 1.0]) - apply(h, points)
    return np.sqrt(np.abs(dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0]))


def _match(holes, h, markers, radius_share):
    """Holes paired with the markers `h` predicts them at. -> (marker indices,
    hole indices, the markers' spacing on the frame)."""
    predicted = apply(h, markers.points)
    if len(predicted) > 1:
        gaps = np.sqrt(((predicted[:, None] - predicted[None]) ** 2).sum(-1))
        np.fill_diagonal(gaps, np.inf)
        spacing = float(np.median(gaps.min(axis=1)))
    else:
        spacing = float(max(markers.width, markers.height))
    if not len(holes):
        return np.zeros(0, int), np.zeros(0, int), spacing
    scale = _local_scale(h, markers.points)
    expected = (2.0 * markers.half * scale) ** 2 * markers.fill
    distance = np.sqrt(((predicted[:, None] - holes[None, :, :2]) ** 2).sum(-1))
    ratio = holes[None, :, 2] / np.maximum(expected[:, None], 1e-6)
    fits = (distance <= radius_share * spacing) & (ratio >= MATCH_AREA[0]) & (ratio <= MATCH_AREA[1])
    distance = np.where(fits, distance, np.inf)
    chosen_m, chosen_h = [], []
    # Nearest first, each hole and each marker used once.
    order = np.dstack(np.unravel_index(np.argsort(distance, axis=None), distance.shape))[0]
    taken_m, taken_h = set(), set()
    for m, k in order:
        if not np.isfinite(distance[m, k]):
            break
        if m in taken_m or k in taken_h:
            continue
        taken_m.add(m)
        taken_h.add(k)
        chosen_m.append(m)
        chosen_h.append(k)
    return np.array(chosen_m, int), np.array(chosen_h, int), spacing


def solve(holes, predicted, markers, rng=None):
    """The plane of this frame's markers, starting from the one predicted for it.
    -> (3x3 tracker -> image, markers agreeing) or (None, 0)."""
    rng = rng or np.random.default_rng(0)
    h = predicted
    agreeing = 0
    for radius in (MATCH_RADIUS, MATCH_RADIUS * 0.6):
        m, k, spacing = _match(holes, h, markers, radius)
        if len(m) < MIN_MATCHES:
            return None, 0
        src, dst = markers.points[m], holes[k, :2]
        tolerance = max(INLIER_PX, INLIER_SHARE * spacing)
        best = None
        # The prediction itself is the first hypothesis: on a frame the plane
        # has hardly moved in, it is already the answer.
        candidates = [h]
        for _ in range(RANSAC_ROUNDS):
            pick = rng.choice(len(m), 4, replace=False)
            candidate = fit_homography(src[pick], dst[pick])
            if candidate is not None:
                candidates.append(candidate)
        for candidate in candidates:
            with np.errstate(divide="ignore", invalid="ignore"):
                error = np.sqrt(((apply(candidate, src) - dst) ** 2).sum(-1))
            error = np.where(np.isfinite(error), error, np.inf)
            inliers = error <= tolerance
            if best is None or inliers.sum() > best.sum():
                best = inliers
        if best.sum() < MIN_MATCHES:
            return None, 0
        refined = fit_homography(src[best], dst[best])
        if refined is None:
            return None, 0
        h, agreeing = refined, int(best.sum())
    return h, agreeing


def shift_sides(corners, offsets):
    """A quad with each side moved out along its own normal by `offsets[k]`
    pixels (in when negative), corners re-intersected. Tracker order kept."""
    centre = corners.mean(axis=0)
    lines = []
    for side in range(4):
        a, b = corners[side], corners[(side + 1) % 4]
        direction = (b - a) / max(float(np.linalg.norm(b - a)), 1e-9)
        normal = np.array([-direction[1], direction[0]])
        if np.dot(a - centre, normal) < 0:
            normal = -normal
        lines.append((a + float(offsets[side]) * normal, direction))
    out = []
    for side in range(4):
        corner = _intersect(lines[side - 1], lines[side])
        out.append(corners[side] if corner is None else corner)
    return np.array(out)


def refine_sides(outline, corners, max_turn=None):
    """Each side of a quad moved onto the key's outline, where the outline has
    one there. -> (corners, each side's move out along its normal in pixels —
    NaN where the outline did not support it).

    The markers say where the screen's plane is; H3 draws them near, not exactly
    at, where the tracker has them, and a plane through them puts the screen's
    edges a pixel or three off its glass — a strip of tracker along one side
    and the content over the bezel along the other. The outline is exact where
    it is the screen's edge. It is not where something is in front of the edge,
    and there it is far from the side and sparse along it: a side is only moved
    when enough of its length has outline right beside it — and, given
    `max_turn` (degrees), only when the line found is within that of the side
    it was looked for from.
    """
    outline = np.asarray(outline, dtype=np.float64)
    moved = np.full(4, np.nan)
    if not len(outline):
        return corners, moved
    centre = corners.mean(axis=0)
    margin = (1.0 - SIDE_MIDDLE) / 2.0
    lines = []
    for side in range(4):
        a, b = corners[side], corners[(side + 1) % 4]
        length = float(np.linalg.norm(b - a))
        direction = (b - a) / max(length, 1e-9)
        normal = np.array([-direction[1], direction[0]])
        rel = outline - a
        along = rel @ direction / max(length, 1e-9)
        across = np.abs(rel @ normal)
        near = (along >= margin) & (along <= 1.0 - margin) & (across <= max(3.0, SIDE_BAND * length))
        line = (a, direction)
        if near.sum() >= max(MIN_SIDE_POINTS, SIDE_SUPPORT * length * SIDE_MIDDLE):
            point, fitted = _fit_line(outline[near])
            fitted_normal = np.array([-fitted[1], fitted[0]])
            residual = np.abs((outline[near] - point) @ fitted_normal)
            turn = np.degrees(np.arccos(min(1.0, abs(float(np.dot(fitted, direction))))))
            if np.median(residual) <= SIDE_STRAIGHT and (max_turn is None or turn <= max_turn):
                # No half-pixel nudge outward, unlike `fit_quad`: the key that
                # makes this outline is grown from its core into the screen's
                # dim edge, so its last pixels are already the half-covered ones.
                line = (point, fitted)
                outward = normal if np.dot(a - centre, normal) >= 0 else -normal
                middle = (a + b) / 2.0
                moved[side] = float(np.dot(line[0] - middle, outward))
        lines.append(line)
    out = []
    for side in range(4):
        corner = _intersect(lines[side - 1], lines[side])
        if corner is None:
            return corners, np.full(4, np.nan)
        out.append(corner)
    # Tracker order is kept: side k runs from corner k to corner k + 1, so the
    # corner between sides k - 1 and k is corner k.
    return np.array(out), moved


def luma_small(frame):
    """A frame -> a small grey copy for the cut test."""
    grey = frame @ np.array([0.2126, 0.7152, 0.0722], dtype=frame.dtype)
    step = max(1, min(grey.shape) // 96)
    return grey[::step, ::step].astype(np.float32)


# ---- a shot ---------------------------------------------------------------------------


def _smooth(track):
    """Gaussian over time. (N, 4, 2) -> (N, 4, 2).

    Padded at each end by the motion continued (a point reflection) rather than
    by the end frame repeated: a screen drifting across the frame would
    otherwise be held back at the start and end of every shot by the frames
    that were not there.
    """
    if len(track) < 2:
        return track
    offsets = np.arange(-SMOOTH_RADIUS, SMOOTH_RADIUS + 1)
    kernel = np.exp(-0.5 * (offsets / SMOOTH_SIGMA) ** 2)
    kernel /= kernel.sum()
    reach = min(SMOOTH_RADIUS, len(track) - 1)
    head = 2 * track[0] - track[1:reach + 1][::-1]
    tail = 2 * track[-1] - track[-reach - 1:-1][::-1]
    head = np.concatenate([np.repeat(head[:1], SMOOTH_RADIUS - reach, axis=0), head])
    tail = np.concatenate([tail, np.repeat(tail[-1:], SMOOTH_RADIUS - reach, axis=0)])
    padded = np.concatenate([head, track, tail])
    out = np.zeros_like(track)
    for k, weight in zip(range(2 * SMOOTH_RADIUS + 1), kernel):
        out += weight * padded[k:k + len(track)]
    return out


@dataclass
class Shot:
    start: int
    stop: int
    corners: np.ndarray | None     # (stop - start, 4, 2) in tracker order, or None
    seen: int                      # frames the screen was solved in
    solved: np.ndarray | None = None   # (stop - start,) bool: solved, or filled in


def upright_score(corners):
    """How upright a tracker-ordered quad reads: its top edge's midpoint above
    its bottom edge's, in image y (down)."""
    top = (corners[0] + corners[1]) / 2.0
    bottom = (corners[2] + corners[3]) / 2.0
    return float(bottom[1] - top[1])


def _turn(found, symmetric, aspect=None):
    """The quarter turn a starting fit's markers line up best under.

    With no markers (`aspect` given, the tracker's width over height) every turn
    scores alike, and the fit's own shape decides between the two readings that
    match it — a phone is tall until it is on its side — then the upright one.
    """
    best = found.turns.max()
    ties = [t for t in range(4) if found.turns[t] >= best * 0.9]
    if aspect is not None:
        def shape(t):
            c = turned(found.corners, t)
            wide = np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3])
            tall = np.linalg.norm(c[3] - c[0]) + np.linalg.norm(c[2] - c[1])
            return abs(np.log(wide / max(tall, 1e-9)) - np.log(aspect))
        closest = min(shape(t) for t in ties)
        ties = [t for t in ties if shape(t) <= closest + 0.05]
        return max(ties, key=lambda t: upright_score(turned(found.corners, t)))
    if symmetric and len(ties) > 1:
        return max(ties, key=lambda t: upright_score(turned(found.corners, t)))
    return int(np.argmax(found.turns))


class _Follow:
    def __init__(self, seen, screen, diagonal):
        self.seen = seen
        self.markers = Markers(screen)
        self.canonical = canonical_corners(screen.width, screen.height)
        self.symmetric = screen.pattern == "plus"
        self.plain = screen.pattern == "none"
        self.aspect = screen.width / screen.height if self.plain else None
        self.cut = CUT_PX * diagonal / np.hypot(1280.0, 720.0)
        self.rng = np.random.default_rng(0)
        self.offsets = {}
        # Plain screens: the frames all four edges were seen in. Motion is only
        # carried forward from those — an edge that was held rather than seen
        # has no motion of its own, and extrapolating it walks it off the glass.
        self.whole = {}

    def measure(self, index, predicted):
        """This frame's plane, from where it is predicted: solved from its
        markers, or — with no markers — from its edges alone. -> 3x3, or None.

        With markers, how far each edge sits from the markers' plane is noted
        (`offsets`) rather than applied: H3 draws the tracker a little larger or
        smaller than the glass, by the same amount all through a shot, and the
        shot's median of it is what `_shot` puts back. One frame's edge is only
        one vote — an arm lined up with a side cannot move it.
        """
        seen = self.seen[index]
        if self.plain:
            corners, moved = refine_sides(seen.outline, self.corners(predicted), PLAIN_TURN)
            sides = int(np.isfinite(moved).sum())
            self.whole[index] = sides == 4
            return self.plane(corners) if sides >= PLAIN_SIDES else None
        h, _ = solve(seen.holes, predicted, self.markers, self.rng)
        if h is not None:
            self.offsets[index] = refine_sides(seen.outline, self.corners(h))[1]
        return h

    def corners(self, h):
        return apply(h, self.canonical)

    def plane(self, corners):
        return fit_homography(self.canonical, corners)

    def begin(self, index):
        """A plane off the starting fit at `index`, sharpened by its markers."""
        found = self.seen[index].start
        h = self.plane(turned(found.corners, _turn(found, self.symmetric, self.aspect)))
        measured = self.measure(index, h)
        return measured if measured is not None else h

    def restart(self, index, predicted):
        """The starting fit at `index`, turned to agree with `predicted`."""
        found = self.seen[index].start
        target = self.corners(predicted)
        turn = min(range(4), key=lambda t: np.abs(turned(found.corners, t) - target).max())
        return self.plane(turned(found.corners, turn))

    def run(self, frames, planes, first):
        """Follow the markers over `frames` (in order, either direction), from
        the plane `first`. Fills `planes`; -> the frame a cut was found at, or
        None."""
        history = [first]
        coast = 0
        previous = None
        for index in frames:
            last = self.corners(history[-1])
            moving = not self.plain or (self.whole.get(previous, False)
                                        and self.whole.get(previous - (index - previous), False))
            if len(history) > 1 and coast == 0 and moving:
                velocity = last - self.corners(history[-2])
                predicted = self.plane(last + velocity)
            else:
                predicted = history[-1]
            if predicted is None:
                predicted = history[-1]
            h = self.measure(index, predicted)
            # A plain screen has no markers to be found somewhere else, and a
            # starting fit on a covered one is the uncovered part of it: there
            # it only ever coasts, and only a change of picture cuts.
            if h is None and self.seen[index].start is not None and not self.plain:
                fresh = self.restart(index, predicted)
                expected = self.corners(predicted)
                limit = max(self.cut, CUT_SHARE * float(np.linalg.norm(expected[2] - expected[0])))
                if np.abs(self.corners(fresh) - expected).max() > limit:
                    # Lost where it was, found somewhere else: a new shot.
                    return index
                h = fresh
            planes[index] = h
            previous = index
            if h is None:
                coast += 1
                if coast > MAX_COAST:
                    history = history[-1:]
                continue
            coast = 0
            history.append(h)
        return None

    def shots(self, a, b):
        starts = [i for i in range(a, b) if self.seen[i].start is not None
                  and self.seen[i].start.turns.max() >= SEED_SCORE]
        if not starts:
            return [Shot(a, b, None, 0)]
        first = max(starts, key=lambda i: self.seen[i].start.turns.max())
        planes = {first: self.begin(first)}
        stop = self.run(range(first + 1, b), planes, planes[first]) or b
        back = self.run(range(first - 1, a - 1, -1), planes, planes[first])
        start = a if back is None else back + 1
        out = []
        if start > a:
            out += self.shots(a, start)
        out.append(self._shot(start, stop, planes))
        if stop < b:
            out += self.shots(stop, b)
        return out

    def _shot(self, start, stop, planes):
        solved = [i for i in range(start, stop) if planes.get(i) is not None]
        frames = np.arange(start, stop)
        raw = np.array([self.corners(planes[i]) for i in solved])
        # Each side put back on the glass by the shot's median of how far the
        # edge sat from the markers' plane — where enough frames saw that edge.
        offsets = np.array([self.offsets[i] for i in solved if i in self.offsets]).reshape(-1, 4)
        if len(offsets):
            seen_enough = np.isfinite(offsets).sum(axis=0) >= max(3, EDGE_VOTES * len(solved))
            shift = np.where(seen_enough, np.nanmedian(np.where(np.isfinite(offsets), offsets, np.nan),
                                                       axis=0), 0.0)
            shift = np.nan_to_num(shift)
            raw = np.array([shift_sides(c, shift) for c in raw])
        track = np.empty((stop - start, 4, 2))
        for corner in range(4):
            for axis in range(2):
                # Unsolved frames from their neighbours in the shot; past
                # either end, the nearest one held.
                track[:, corner, axis] = np.interp(frames, solved, raw[:, corner, axis])
        mask = np.isin(frames, solved)
        return Shot(start, stop, _smooth(track), len(solved), mask)


def follow(seen, screen, diagonal):
    """Every frame's `Seen` -> one `Shot` per shot, corners in tracker order."""
    lumas = [s.luma for s in seen]
    bounds = [0] + [i for i in range(1, len(seen))
                    if float(np.abs(lumas[i] - lumas[i - 1]).mean()) > CUT_LUMA] + [len(seen)]
    follower = _Follow(seen, screen, diagonal)
    shots = []
    for a, b in zip(bounds, bounds[1:]):
        shots += follower.shots(a, b)
    return shots
