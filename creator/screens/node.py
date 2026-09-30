"""The screen pass as nodes: one over a finished reel, one over a still.

**The reel node runs last, after ReDetail and the DLSS 5 refiner.** Both of
those re-draw the frame, and either would soften text that is meant to be the
source file's own. Last also means the content goes in at the size the file
leaves at: a render that upscales composites at the upscaled size, which is
what the spec's own 2x plate was for — the pass does not scale on its own,
because every part of a reel has to come out one size (`mux.reel_geometry`)
and a pass that doubled only the parts with screens would break the strip.

It runs at the end rather than inline for a second reason. A seam hands the
next segment the pass's last frames, and a next segment that continues a shot
with a screen in it is shown a tracker in its own reference list; handing it
the replaced content instead would put the content through the model after all.

**Two walks over each pass.** The first tracks every frame and keeps only the
corners, so the shots, the turn and the smoothing can be decided over the whole
pass before anything is drawn. The second composites a chunk at a time and
writes through `spill.rewrite`, which keeps the pass's sound file as it is.

**Failure is said, never shipped quietly.** A frame the screen is lost in takes
its corners from its neighbours in the same shot; a shot the screen is never
found in passes through untouched; and a pass where most frames failed keeps
the raw render for that screen, with the reason on the node and in the log.
"""

import json
import logging

import numpy as np
import torch

import comfy.model_management
import comfy.utils
from comfy_api.latest import io

from .. import media, spill
from ..timeline import REEL_TYPE
from . import composite, spec, track, tracker

PASS_NODE = "ContinuityScreenPass"
STILL_NODE = "ContinuityScreenStill"

CHUNK = 16
# The share of a pass's frames the screen has to be found in, or the pass keeps
# its raw render for that screen.
MIN_SEEN = 0.5
# The content is resampled to about this many times the largest size the
# screen reaches on the frame, and never past the tracker's own size.
CONTENT_OVERSAMPLE = 2.0

log = logging.getLogger(__name__)


def _screens(raw):
    return [spec.Screen.from_json(entry) for entry in raw]


def _report(node_id, text):
    log.warning("continuity: %s", text)
    if not node_id:
        return
    try:
        from server import PromptServer
        server = getattr(PromptServer, "instance", None)
        if server is not None:
            server.send_progress_text(text, node_id)
    except Exception:  # noqa: BLE001 — a message that cannot be shown is still in the log
        pass


def _as_float(frame):
    return np.asarray(frame, dtype=np.float32) / 255.0


class Source:
    """One screen's content: a picture, or a clip's frames, fitted lazily."""

    def __init__(self, screen, span):
        self.screen = screen
        scale = min(1.0, CONTENT_OVERSAMPLE * span / max(screen.width, screen.height))
        self.width = max(16, int(round(screen.width * scale)))
        self.height = max(16, int(round(screen.height * scale)))
        if screen.kind == "image":
            picture = media.load_image(screen.filename)[0]
            self.frames = [(picture.numpy() * 255.0).round().astype(np.uint8)]
            self.fps = None
        else:
            frames, _ = media.load_video(screen.filename)
            self.frames = (frames.numpy() * 255.0).round().astype(np.uint8)
            self.fps = float(media.TARGET_FPS)
        self._index = None
        self._content = None

    def index_at(self, seconds):
        """Which content frame shows at `seconds` into the screen's clock."""
        if self.fps is None:
            return 0
        count = len(self.frames)
        at = int(np.floor((self.screen.offset_s + seconds) * self.fps + 1e-6))
        if at < count:
            return max(0, at)
        if self.screen.when_short == "loop":
            return at % count
        if self.screen.when_short == "first":
            return 0
        return count - 1

    def at(self, seconds):
        index = self.index_at(seconds)
        if index != self._index:
            self._content = composite.Content(self.frames[index], self.width, self.height,
                                              self.screen.fit)
            self._index = index
        return self._content


def track_pass(frames, screen, tick=None):
    """Every frame of a pass -> (shots, frames the screen was solved in)."""
    answer = track.Answer(screen.pattern, screen.colour, screen.width, screen.height)
    seen = []
    for index in range(len(frames)):
        comfy.model_management.throw_exception_if_processing_interrupted()
        seen.append(track.observe(_as_float(frames[index]), screen, answer))
        if tick:
            tick()
    height, width = frames.shape[1], frames.shape[2]
    shots = track.follow(seen, screen, float(np.hypot(width, height)))
    return shots, sum(shot.seen for shot in shots)


def _largest_side(shots):
    """The longest side the screen reaches anywhere in the pass, in pixels."""
    sides = [np.linalg.norm(np.roll(shot.corners, -1, axis=1) - shot.corners, axis=2).max()
             for shot in shots if shot.corners is not None]
    return max(sides, default=1.0)


def plan(frames, screens, node_id, where, tick=None):
    """Track each screen over a pass. -> [(screen, shots, source)] for the screens
    worth drawing; the rest are reported and left raw."""
    count = len(frames)
    drawn = []
    for number, screen in enumerate(screens, start=1):
        shots, seen = track_pass(frames, screen, tick)
        if seen < MIN_SEEN * count:
            _report(node_id, f"{where}: @{screen.handle} was found in only {seen} of "
                             f"{count} frames, so it is left as rendered. Check that "
                             f"the {spec.COLOUR[screen.colour].label.lower()} tracker is "
                             f"visible and not too small in the shot.")
            continue
        lost = [shot for shot in shots if shot.corners is None]
        if lost:
            log.info("continuity: %s screen %d: %d of %d shots show no screen and "
                     "pass through", where, number, len(lost), len(shots))
        drawn.append((screen, shots, Source(screen, _largest_side(shots))))
    return drawn


def _at(shots, index):
    """-> (the shot holding frame `index`, its corners there or None, solved?)."""
    shot = next(s for s in shots if s.start <= index < s.stop)
    if shot.corners is None:
        return shot, None, False
    offset = index - shot.start
    return shot, shot.corners[offset], bool(shot.solved[offset]) if shot.solved is not None else True


def draw(frame, index, fps, drawn, holes, seed, debug=None):
    """Every screen onto one frame (float, changed in place). With `debug` (the
    raw frame), -> (frame, the debug view of it) instead."""
    raw = frame.copy() if debug else None
    views = []
    for (screen, shots, source), answer in zip(drawn, holes):
        shot, corners, solved = _at(shots, index)
        made = []
        if corners is not None:
            clock = (index - shot.start) if screen.cuts == "restart" else index
            rng = np.random.default_rng((seed, index, screen.width, screen.height))
            composite.compose(frame, screen, corners, source.at(clock / fps), answer, rng,
                              drawn=made)
        views.append((screen, corners, solved, made[0] if made else None))
    if not debug:
        return frame
    view = raw * 0.55
    for screen, corners, solved, made in views:
        view = np.maximum(view, composite.overlay(raw, screen, corners, solved, made)) \
            if corners is not None else view
    return frame, view


def replace_pass(source, screens, seed, node_id, where, tick, debug=False):
    """One pass with its screens replaced. -> (its spec, the debug view's spec or
    None). The debug view is a second walk over the frames rather than a second
    stream out of the first, so the pass never holds more than a chunk of
    either."""
    frames = spill.open_frames(source)
    fps = float(source.get("fps") or 24.0)
    drawn = plan(frames, screens, node_id, where, tick)
    if not drawn:
        return source, None
    holes = [composite.answer_holes(screen) for screen, _, _ in drawn]
    count = int(source["frames"])

    def blocks(view):
        for start in range(0, count, CHUNK):
            stop = min(start + CHUNK, count)
            block = np.empty((stop - start, frames.shape[1], frames.shape[2], 3),
                             dtype=np.float32)
            for offset in range(stop - start):
                comfy.model_management.throw_exception_if_processing_interrupted()
                out = draw(_as_float(frames[start + offset]), start + offset,
                           fps, drawn, holes, seed, debug=view)
                block[offset] = out[1] if view else out
                if not view:
                    tick()
            yield torch.from_numpy(block)

    replaced = spill.rewrite(source, blocks(False))
    return replaced, (spill.rewrite(source, blocks(True)) if debug else None)


class ContinuityScreenPass(io.ComfyNode):
    """Every screen on every generated pass replaced with its content."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=PASS_NODE,
            display_name="Continuity Screen Pass",
            category="Continuity/internal",
            description="Finds each screen's tracker in every frame of the passes "
                        "that carry one and puts the screen's real content on it. "
                        "Written into the graph by render.emit.",
            is_dev_only=True,
            inputs=[
                io.Custom(REEL_TYPE).Input("reel"),
                io.String.Input("screens_data", multiline=True,
                                tooltip="{part index: [screen, ...]} — written by the "
                                        "render loop."),
                io.Int.Input("seed", default=0, min=0, max=0x7fffffff),
                io.String.Input("report_to", default="",
                                tooltip="The node whose face a warning is shown on."),
                io.Boolean.Input("debug", default=False,
                                 tooltip="Also hand out what the pass saw, drawn over the "
                                         "raw render, as a second reel."),
            ],
            outputs=[io.Custom(REEL_TYPE).Output(display_name="reel"),
                     io.Custom(REEL_TYPE).Output(display_name="debug")],
        )

    @classmethod
    def execute(cls, reel, screens_data, seed, report_to="", debug=False) -> io.NodeOutput:
        parts = list(reel or [])
        by_part = {int(k): _screens(v) for k, v in json.loads(screens_data or "{}").items()}
        for screens in by_part.values():
            tracker.ensure_all(screens)
        work = [(index, screens) for index, screens in sorted(by_part.items())
                if index < len(parts) and "pass" in parts[index]]
        total = sum(int(parts[i]["pass"]["frames"]) * (len(s) + 1) for i, s in work)
        bar = comfy.utils.ProgressBar(max(1, total))
        done = [0]

        def tick():
            done[0] += 1
            bar.update_absolute(done[0], total)

        out, view = list(parts), list(parts)
        for index, screens in work:
            where = f"part {index + 1}"
            replaced, seen = replace_pass(parts[index]["pass"], screens, int(seed),
                                          report_to, where, tick, debug=bool(debug))
            out[index] = {"pass": replaced}
            if seen is not None:
                view[index] = {"pass": seen}
        return io.NodeOutput(out, view)


class ContinuityScreenStill(io.ComfyNode):
    """The screen pass on a still: once, with no cuts and no smoothing."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=STILL_NODE,
            display_name="Continuity Screen Replace",
            category="Continuity",
            description="Finds screen trackers in a picture and puts real content on "
                        "them. The trackers are the ones Continuity draws; the "
                        "screens list says which, and what each one shows.",
            inputs=[
                io.Image.Input("image"),
                io.String.Input("screens_data", multiline=True,
                                tooltip="[screen, ...] as the Creator writes it."),
                io.Int.Input("seed", default=0, min=0, max=0x7fffffff),
                io.String.Input("report_to", default=""),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image, screens_data, seed, report_to="") -> io.NodeOutput:
        screens = _screens(json.loads(screens_data or "[]"))
        tracker.ensure_all(screens)
        batch = (image.detach().to("cpu").float().clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
        out = np.empty(batch.shape, dtype=np.float32)
        for index in range(batch.shape[0]):
            single = batch[index:index + 1]
            drawn = plan(single, screens, report_to, "this picture")
            holes = [composite.answer_holes(screen) for screen, _, _ in drawn]
            out[index] = draw(_as_float(single[0]), 0, 24.0, drawn, holes, int(seed))
        return io.NodeOutput(torch.from_numpy(out))


# ---- writing them into graphs ------------------------------------------------------


def emit(graph, reel, by_part, seed, report_to, debug=False):
    """The pass over a finished reel. `by_part` is `{part index: (Screen, ...)}`.
    -> the node; its output 0 is the replaced reel, output 1 the debug view."""
    data = {str(k): [s.to_json() for s in v] for k, v in sorted(by_part.items()) if v}
    return graph.node(PASS_NODE, reel=reel,
                      screens_data=json.dumps(data, sort_keys=True),
                      seed=int(seed) % 0x7fffffff, report_to=str(report_to or ""),
                      **({"debug": True} if debug else {}))


def emit_still(graph, image, screens, seed, report_to=""):
    """The still node between a decode and its save. -> the image link."""
    tracker.ensure_all(screens)
    return graph.node(STILL_NODE, image=image,
                      screens_data=json.dumps([s.to_json() for s in screens], sort_keys=True),
                      seed=int(seed) % 0x7fffffff, report_to=str(report_to or "")).out(0)


NODES = [ContinuityScreenPass, ContinuityScreenStill]
