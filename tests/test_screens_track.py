"""The screen pass's arithmetic, on frames whose answer is known.

    <comfy-venv>/bin/python3 tests/test_screens_track.py

A tracker is warped into a synthetic frame at corners chosen here, and the
tracker has to find them again: to a fraction of a pixel, the right way up when
the phone is on its side, past a brighter white wall that has no markers, and
split at a cut rather than smoothed across it. Then a magenta screen is
replaced and nothing magenta may be left outside it — the rim the spec's
prototype had to despill.

The regression the spec asked for — the recorded lab clip — is not in the repo;
this stands in for it with frames that can be rebuilt from nothing. Needs
numpy, scipy, torch and Pillow (ComfyUI's own requirements); skips without.
"""

import layout
from harness import FAILURES, check, passed, skip

try:
    import numpy as np
    from scipy import ndimage
    import torch  # noqa: F401
    from PIL import Image  # noqa: F401
except ImportError as exc:
    skip(f"needs ComfyUI's numeric stack ({exc.name})")

passed("the tracker finds a warped screen to a fraction of a pixel, reads its turn, "
       "ignores a markerless wall, splits at a cut, and a replaced screen leaves no rim")

_pkg = layout.load("screen_spec", "screen_tracker", "screen_track", "screen_composite")
spec, tracker, track, composite = (_pkg.screen_spec, _pkg.screen_tracker,
                                   _pkg.screen_track, _pkg.screen_composite)

H, W = 720, 960


def screen_of(pattern, colour, device="phone"):
    return spec.parse([{"device": device, "content": {"filename": "x.png"},
                        "tracker": {"pattern": pattern, "colour": colour}}])[0]


def scene(seed=0):
    """A mid-grey room with texture and a white wall bigger than any screen."""
    rng = np.random.default_rng(seed)
    frame = np.full((H, W, 3), 0.42, dtype=np.float32)
    frame += rng.normal(0, 0.02, frame.shape).astype(np.float32)
    frame[40:700, 690:950] = 0.93          # the wall: bright, neutral, no markers
    return np.clip(frame, 0, 1)


def place(frame, screen, corners):
    """The screen's tracker drawn onto `corners` (tracker order), antialiased."""
    picture = tracker.draw(screen.pattern, screen.colour, screen.width, screen.height)
    content = composite.Content(picture, screen.width // 3, screen.height // 3, "stretch")
    box = composite.box_around(corners, frame.shape[:2], 2)
    y0, y1, x0, x1 = box
    drawn = composite.warp(content, corners, box)
    cover = composite.polygon(composite.image_order(corners - [x0, y0]), (y1 - y0, x1 - x0))
    frame[y0:y1, x0:x1] = frame[y0:y1, x0:x1] * (1 - cover[..., None]) + drawn * cover[..., None]
    return frame


def err(found, truth):
    return float(np.abs(found - truth).max())


# ---- one frame ---------------------------------------------------------------------

phone = screen_of("grid", "white")
answer = track.Answer(phone.pattern, phone.colour, phone.width, phone.height)
# A phone held a little turned and tilted: corners in tracker order, TL TR BR BL.
truth = np.array([[183.3, 141.7], [372.6, 158.2], [351.9, 566.4], [158.4, 548.1]])
frame = place(scene(), phone, truth)
found = track.locate(frame, phone, answer)
check("the screen is found beside a brighter, markerless wall", found is not None, True)
if found is not None:
    turn = int(np.argmax(found.turns))
    check("upright, its corners come back in tracker order", turn, 0)
    check("to within half a pixel", err(track.turned(found.corners, turn), truth) < 0.5, True)

# The same phone on its side: tracker top-left now at the image's top right.
side = np.roll(np.array([[150.0, 180.0], [560.0, 170.0], [566.0, 370.0], [156.0, 380.0]]), 1, axis=0)
turned_frame = place(scene(1), phone, side)
found = track.locate(turned_frame, phone, answer)
check("a phone on its side is found", found is not None, True)
if found is not None:
    turn = int(np.argmax(found.turns))
    check("and the triangles say which way is up",
          err(track.turned(found.corners, turn), side) < 0.75, True)

check("an empty room has no screen in it", track.locate(scene(2), phone, answer), None)

# ---- a shot, and a cut --------------------------------------------------------------------

diagonal = float(np.hypot(W, H))


def followed(frames, screen):
    answer_ = track.Answer(screen.pattern, screen.colour, screen.width, screen.height)
    return track.follow([track.observe(f, screen, answer_) for f in frames], screen, diagonal)


def corners_at(shots, index):
    shot = next(s for s in shots if s.start <= index < s.stop)
    return None if shot.corners is None else shot.corners[index - shot.start]


frames, truths = [], []
for index in range(12):
    drift = np.array([index * 1.5, index * 0.5])
    at = truth + drift if index < 6 else truth + [260.0, 60.0] + drift
    truths.append(at)
    frames.append(place(scene(10 + index), phone, at))
shots = followed(frames, phone)
check("the jump is a cut, and the drift is not", [(s.start, s.stop) for s in shots], [(0, 6), (6, 12)])
check("every frame of both shots is solved", [s.seen for s in shots], [6, 6])
worst = max(err(corners_at(shots, k), truths[k]) for k in range(12))
check("the plane stays within a pixel of the truth on both sides of the cut", worst < 1.0, True)

# ---- somebody walks in front of it -----------------------------------------------------------
#
# The case the first version got wrong on the lab: a TV on a wall, and a person
# crossing in front of it — covering a side, then most of it, then the other
# side. The screen does not move; where it is has to come out the same in
# every frame, and nothing may be drawn on the person.

tv = screen_of("grid", "white", device="tv")
tv_truth = np.array([[300.0, 180.0], [620.0, 188.0], [616.0, 372.0], [296.0, 366.0]])
walk, bodies = [], []
for index in range(16):
    frame = place(scene(40), tv, tv_truth)
    x = 180 + index * 34                      # the body's left edge, crossing right
    body = np.zeros((H, W), dtype=bool)
    body[120:720, x:x + 170] = True
    frame[body] = (0.12, 0.1, 0.09)           # dark clothes
    walk.append(frame)
    bodies.append(body)
shots = followed(walk, tv)
check("a person crossing is not a cut", len(shots), 1)
shot = shots[0]
check("the screen is solved while half of it is covered", int(shot.seen) >= 12, True)
worst = max(err(shot.corners[k], tv_truth) for k in range(16))
check("and where it is stays put, within a pixel, in every frame", worst < 1.0, True)

content = composite.Content(np.full((64, 64, 3), (20, 120, 220), dtype=np.uint8),
                            tv.width // 4, tv.height // 4, "fill")
painted_on_person = 0
tracker_left = 0
for index in (4, 6, 8, 10):
    raw = walk[index]
    out = composite.compose(raw.copy(), tv, shot.corners[index], content,
                            composite.answer_holes(tv), np.random.default_rng(index))
    person_on_screen = bodies[index] & (composite.polygon(composite.image_order(tv_truth), (H, W)) > 0.99)
    # Well inside the body, away from its edge.
    inner = person_on_screen & ndimage.binary_erosion(bodies[index], iterations=3)
    painted_on_person += int((np.abs(out[inner] - raw[inner]).max(axis=-1) > 0.02).sum())
    tracker_left += int(track.key(out, "white")[composite.polygon(composite.image_order(tv_truth), (H, W)) > 0.99].sum())
check("nothing is painted on the person in front of the screen", painted_on_person, 0)
check("and no tracker is left showing beside them", tracker_left, 0)

# The same walk past a plain green TV: no markers, so it is followed by its
# edges — each frame needs three of them clear, and the rest are filled in.
green_tv = screen_of("none", "green", device="tv")
walk_green = []
for index in range(16):
    frame = place(scene(40), green_tv, tv_truth)
    frame[bodies[index]] = (0.12, 0.1, 0.09)
    walk_green.append(frame)
shots = followed(walk_green, green_tv)
check("a plain screen: the person crossing is not a cut", len(shots), 1)
worst = max(err(shots[0].corners[k], tv_truth) for k in range(16))
check("a plain screen stays within a pixel and a half of where it is", worst < 1.5, True)
painted = 0
for index in (4, 8, 12):
    raw = walk_green[index]
    out = composite.compose(raw.copy(), green_tv, shots[0].corners[index], content,
                            composite.answer_holes(green_tv), np.random.default_rng(index))
    inner = bodies[index] & ndimage.binary_erosion(bodies[index], iterations=3) \
        & (composite.polygon(composite.image_order(tv_truth), (H, W)) > 0.99)
    painted += int((np.abs(out[inner] - raw[inner]).max(axis=-1) > 0.02).sum())
    check(f"no green left on the plain screen at frame {index}",
          int(track.key(out, "green")[composite.polygon(composite.image_order(tv_truth), (H, W)) > 0.99].sum()), 0)
check("and nothing painted on the person", painted, 0)

# ---- the composite ---------------------------------------------------------------------------

magenta = screen_of("plus", "magenta")
frame = place(scene(3), magenta, truth)
keyed_before = track.key(frame, "magenta").sum()
content = composite.Content(np.full((64, 32, 3), (20, 120, 220), dtype=np.uint8),
                            magenta.width // 4, magenta.height // 4, "fill")
out = composite.compose(frame.copy(), magenta, truth, content,
                        composite.answer_holes(magenta), np.random.default_rng(0))
check("the magenta screen was there to replace", keyed_before > 20000, True)
check("and not one magenta pixel is left anywhere", int(track.key(out, "magenta").sum()), 0)
centre = out[int(truth[:, 1].mean()), int(truth[:, 0].mean())]
check("the screen shows the content's colour", bool(np.abs(centre - [20 / 255, 120 / 255, 220 / 255]).max() < 0.08),
      True)
far = (40, 900)
check("a pixel far from the screen is untouched", bool(np.allclose(out[far], frame[far])), True)

# ---- the debug view ---------------------------------------------------------------------------

for pattern, colour in (("grid", "white"), ("plus", "magenta"), ("none", "green")):
    screen = screen_of(pattern, colour, device="tv")
    raw = place(scene(5), screen, tv_truth)
    made = []
    composite.compose(raw.copy(), screen, tv_truth, content, composite.answer_holes(screen),
                      np.random.default_rng(0), drawn=made)
    view = composite.overlay(raw, screen, tv_truth, True, made[0] if made else None)
    check(f"the debug view draws for the {pattern} pattern", view.shape, raw.shape)
