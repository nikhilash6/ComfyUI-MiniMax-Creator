"""What a screen is, before anything is drawn or tracked. Pure Python.

`compile.py` reads this before a loader exists, so it imports nothing but the
standard library. The frontend mirrors the tables and the validation in
`web/creator/screens.js`; Python is authoritative and `test_screens_mirror.py`
holds the two together.

**The blob says what the screen shows; the settings say what the tracker looks
like.** A card's `screens` list carries the device, the content file and how it
is fitted and timed — what the piece *is*. The tracker's colour and pattern are
this machine's preference (`settings.screen_tracker`) and are stamped onto each
screen by the node before compiling (`stamp`), which puts them in the segment's
cache key: a tracker changed on the settings page is a different reference
picture, and the render has to be made again for it.

**Why plain green is the default.** Tried on the lab, 2026-09-30. The spec's
magenta threw a pink cast on the thumb and bezel, and H3 drew faint patterns
inside it that a shading pass carries onto the content. White came back flat
and lit the hand like a lit screen, but every marker pattern left something
behind somewhere: H3 draws the markers near, not exactly on, where the tracker
has them, and their outlines and slivers flickered through the content. A
plain green screen, the way one is shot on set, has nothing on it to draw off,
and on the lab it came out cleanest. Its cost is occlusion: with no markers it
is followed by its edges alone, so a frame where something covers more than
one of them is filled in from the frames around it. The marker patterns stay
on offer for shots where something crosses the screen for long.
"""

import math
import re
from dataclasses import dataclass

# ---- the devices --------------------------------------------------------------
#
# Each type's tracker is drawn at its screen's own aspect, so the model is shown
# a picture shaped like the screen it is asked to put it on. `noun` is what the
# prompt calls it.

@dataclass(frozen=True)
class Device:
    id: str
    label: str
    noun: str
    width: int
    height: int


DEVICES = (
    Device("phone", "Smartphone", "phone screen", 1080, 2340),
    Device("tablet", "Tablet", "tablet screen", 1536, 2048),
    Device("laptop", "Laptop", "laptop screen", 1920, 1200),
    Device("monitor", "Desktop monitor", "monitor screen", 1920, 1080),
    Device("tv", "TV", "TV screen", 1920, 1080),
    Device("custom", "Custom", "screen", 1920, 1080),
)
DEVICE = {device.id: device for device in DEVICES}
DEFAULT_DEVICE = "phone"

# A custom screen's long side, and how far its aspect may stretch. Past 4:1 the
# tracker's markers stop fitting across the short side.
CUSTOM_LONG_SIDE = 1920
MAX_CUSTOM_RATIO = 4.0

# ---- the tracker ---------------------------------------------------------------
#
# Colours in the order a second screen takes the next free one. `fill` is what is
# drawn; white is drawn a little under full so the model has headroom to shade it
# without clipping. `hue` is the key's centre in degrees, or None for a neutral
# key found by brightness rather than by hue.

@dataclass(frozen=True)
class KeyColour:
    id: str
    label: str
    fill: tuple
    hue: float | None
    words: str


COLOURS = (
    KeyColour("white", "White", (242, 242, 242), None, "white"),
    KeyColour("magenta", "Magenta", (255, 0, 255), 300.0, "magenta"),
    KeyColour("green", "Green", (0, 255, 0), 120.0, "green"),
    KeyColour("blue", "Blue", (0, 0, 255), 240.0, "blue"),
)
COLOUR = {colour.id: colour for colour in COLOURS}

# `grid`: squares and four upward triangles — points for the plane to be solved
# from however much of the screen is covered, and the triangles say which way is
# up when the device turns. `plus`: the spec's four corner crosses and one large
# centre plus, which survives blur best and is symmetric under a half turn.
# `none`: the key colour and nothing else, the way a screen is shot green on
# set — nothing on it for H3 to draw slightly off, so nothing to leave a sliver
# of, and the screen is followed by its four edges alone.
PATTERNS = ("grid", "plus", "none")
PATTERN_WORDS = {
    "grid": "an even grid of small black squares and a small black "
            "upward-pointing triangle in each corner",
    "plus": "small black crosshairs in each corner and a large black plus in the "
            "centre",
    "none": "nothing on it",
}

DEFAULT_STYLE = {"pattern": "none", "colour": "green"}

# ---- what a card may say -------------------------------------------------------

# One screen per key colour would allow four; two is what the pass has been
# measured against, and a second pair of eyes on a scene with three trackers in
# it is a scene nobody has rendered yet.
MAX_SCREENS = 2
FITS = ("fill", "fit", "stretch")
WHEN_SHORT = ("hold", "loop", "first")
CUTS = ("continue", "restart")
KINDS = ("image", "video")
MAX_OFFSET_S = 3600.0

# The handle the tracker rides under in a request's reference list. No `@`
# citation can spell it — `compile.HANDLE_RE` wants a dash and a number — so a
# user's own attachment can never collide with it.
HANDLE_PREFIX = "screen"
TAKE = "screen"

# Where the drawn trackers live: ComfyUI's temp folder, addressed with core's own
# `[temp]` annotation, which the loaders resolve like any other file. Not input/:
# a tracker is the pack's working file, and every one ever drawn used to sit in
# the picker beside the user's own pictures. Temp is emptied when ComfyUI
# starts, and the render draws what it needs before anything reads it
# (`tracker.ensure`). Named by everything that decides the pixels, so within a
# session one is drawn once and a redesign is a new name.
TRACKER_DIR = "continuity_trackers"
TRACKER_FOLDER = "temp"
# 2: the grid pattern went from 3 x 3 to 4 across the short side.
TRACKER_VERSION = 2


class ScreenError(ValueError):
    """A `screens` list that cannot be rendered."""


@dataclass(frozen=True)
class Screen:
    device: str
    width: int                 # the tracker's size, which is the screen's aspect
    height: int
    filename: str              # the content, relative to input/
    kind: str                  # image | video
    fit: str
    offset_s: float
    when_short: str
    cuts: str
    pattern: str
    colour: str

    @property
    def tracker_file(self):
        return tracker_file(self.pattern, self.colour, self.width, self.height)

    @property
    def noun(self):
        return DEVICE[self.device].noun

    @classmethod
    def from_json(cls, data):
        """`to_json`'s dict back into a `Screen`."""
        return cls(**{key: data[key] for key in cls.__dataclass_fields__})

    def to_json(self):
        """What the screen pass is handed — plain data, sorted by the caller."""
        return {"device": self.device, "width": self.width, "height": self.height,
                "filename": self.filename, "kind": self.kind, "fit": self.fit,
                "offset_s": self.offset_s, "when_short": self.when_short,
                "cuts": self.cuts, "pattern": self.pattern, "colour": self.colour}


def handle_for(index):
    """Screen `index`'s reference handle. Screens are told apart by position."""
    return f"{HANDLE_PREFIX}{index + 1}"


def tracker_file(pattern, colour, width, height):
    return (f"{TRACKER_DIR}/v{TRACKER_VERSION}-{pattern}-{colour}-{width}x{height}.png "
            f"[{TRACKER_FOLDER}]")


def tracker_path(name):
    """`tracker_file(...)` -> its path under the folder it lives in, the
    annotation off."""
    return name[:-len(f" [{TRACKER_FOLDER}]")] if name.endswith(f" [{TRACKER_FOLDER}]") else name


def clean_style(raw):
    """A settings value -> `{"pattern", "colour"}`. Raises ValueError."""
    if raw is None:
        return dict(DEFAULT_STYLE)
    if not isinstance(raw, dict):
        raise ValueError("screen_tracker must be an object")
    pattern = raw.get("pattern", DEFAULT_STYLE["pattern"])
    colour = raw.get("colour", DEFAULT_STYLE["colour"])
    if pattern not in PATTERNS:
        raise ValueError(f"screen_tracker pattern must be one of {', '.join(PATTERNS)}")
    if colour not in COLOUR:
        raise ValueError(f"screen_tracker colour must be one of {', '.join(COLOUR)}")
    return {"pattern": pattern, "colour": colour}


def assign_colours(first, count):
    """The key colour of each of `count` screens: the chosen one first, then the
    next free ones in palette order."""
    order = [first] + [c.id for c in COLOURS if c.id != first]
    return order[:count]


def stamp(raw_screens, style):
    """A card's `screens` with this machine's tracker written onto each one.

    Done by the node, off the settings, before compiling — see the module
    docstring. Returns the list unchanged when there is none, so a card without
    screens compiles to the bytes it always did.
    """
    if not isinstance(raw_screens, list) or not raw_screens:
        return raw_screens
    style = clean_style(style)
    colours = assign_colours(style["colour"], len(raw_screens))
    out = []
    for index, screen in enumerate(raw_screens):
        if not isinstance(screen, dict):
            out.append(screen)
            continue
        out.append({**screen, "tracker": {"pattern": style["pattern"],
                                          "colour": colours[min(index, len(colours) - 1)]}})
    return out


def stamp_piece(piece, style):
    """A piece blob with `stamp` applied to every card that has screens. The
    blob itself comes back when none has, so a piece without screens is the
    very object it was."""
    segments = piece.get("segments") if isinstance(piece, dict) else None
    if not isinstance(segments, list) or not any(
            isinstance(seg, dict) and seg.get("screens") for seg in segments):
        return piece
    return {**piece, "segments": [
        {**seg, "screens": stamp(seg["screens"], style)}
        if isinstance(seg, dict) and seg.get("screens") else seg
        for seg in segments]}


def _size(device, aspect):
    if device != "custom":
        return DEVICE[device].width, DEVICE[device].height
    if (not isinstance(aspect, (list, tuple)) or len(aspect) != 2
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in aspect)
            or not all(v > 0 for v in aspect)):
        raise ScreenError("a custom screen needs its aspect as two positive numbers, "
                          "width and height")
    w, h = float(aspect[0]), float(aspect[1])
    if max(w, h) / min(w, h) > MAX_CUSTOM_RATIO:
        raise ScreenError(f"a custom screen can be at most {MAX_CUSTOM_RATIO:g}:1 — "
                          f"the tracker's markers stop fitting past that")
    scale = CUSTOM_LONG_SIDE / max(w, h)
    # Even sides, so the tracker encodes wherever a video encoder is handed it.
    return (max(2, int(round(w * scale / 2)) * 2), max(2, int(round(h * scale / 2)) * 2))


def _content(raw, where):
    content = raw.get("content")
    if not isinstance(content, dict):
        raise ScreenError(f"{where} has nothing to show — pick a picture or a clip for it")
    filename = content.get("filename")
    if not isinstance(filename, str) or not filename.strip():
        raise ScreenError(f"{where} has nothing to show — pick a picture or a clip for it")
    kind = content.get("kind", "image")
    if kind not in KINDS:
        raise ScreenError(f"{where}'s content must be a picture or a clip, not {kind!r}")
    return filename.strip(), kind


def _choice(raw, key, options, default, where):
    value = raw.get(key, default)
    if value not in options:
        raise ScreenError(f"{where}'s {key.replace('_', ' ')} must be one of "
                          f"{', '.join(options)}, not {value!r}")
    return value


def parse(raw_screens, style=None):
    """A card's `screens` value -> a tuple of `Screen`. Raises ScreenError.

    Absent, None and an empty list are all "no screens" and give `()`. A screen
    carries its tracker when the node stamped one; otherwise it takes `style`
    (the default when that is None), coloured in palette order, which is what
    `stamp` would have written.
    """
    if raw_screens is None or raw_screens == []:
        return ()
    if not isinstance(raw_screens, list):
        raise ScreenError("screens must be a list")
    if len(raw_screens) > MAX_SCREENS:
        raise ScreenError(f"a shot can replace at most {MAX_SCREENS} screens")
    fallback = stamp(raw_screens, style or DEFAULT_STYLE)
    screens = []
    for index, raw in enumerate(raw_screens):
        where = f"screen {index + 1}"
        if not isinstance(raw, dict):
            raise ScreenError(f"{where} must be an object")
        device = _choice(raw, "device", tuple(DEVICE), DEFAULT_DEVICE, where)
        width, height = _size(device, raw.get("aspect"))
        filename, kind = _content(raw, where)
        offset = raw.get("offset_s", 0)
        if isinstance(offset, bool) or not isinstance(offset, (int, float)) \
                or not math.isfinite(offset) or not 0 <= offset <= MAX_OFFSET_S:
            raise ScreenError(f"{where}'s start offset must be between 0 and "
                              f"{MAX_OFFSET_S:g} seconds")
        tracker = raw.get("tracker") or fallback[index]["tracker"]
        try:
            tracker = clean_style(tracker)
        except ValueError as exc:
            raise ScreenError(f"{where}: {exc}") from exc
        screens.append(Screen(
            device=device, width=width, height=height, filename=filename, kind=kind,
            fit=_choice(raw, "fit", FITS, FITS[0], where),
            offset_s=round(float(offset), 3),
            when_short=_choice(raw, "when_short", WHEN_SHORT, WHEN_SHORT[0], where),
            cuts=_choice(raw, "cuts", CUTS, CUTS[0], where),
            pattern=tracker["pattern"], colour=tracker["colour"]))
    colours = [s.colour for s in screens]
    if len(set(colours)) != len(colours):
        raise ScreenError("two screens on one shot need two different key colours — "
                          "the pass tells them apart by colour alone")
    return tuple(screens)


def clause(screen, label, ordinal=None):
    """The sentence the prompt carries for one screen.

    It says what the picture on the screen is in so many words, and that it does
    not move, because a model handed a flat colour with no instruction invents a
    UI on it and scrolls it. `ordinal` tells two screens of the same kind apart.
    """
    noun = screen.noun
    if ordinal:
        noun = f"{ordinal} {noun}"
    colour = COLOUR[screen.colour].words
    return (f"The {noun} shows {label} exactly: a flat, uniform {colour} display "
            f"with {PATTERN_WORDS[screen.pattern]}; the screen content stays "
            f"completely static and does not change.")


def clauses(screens, labels):
    """Every screen's sentence, joined. `labels[i]` is screen i's citation."""
    nouns = [s.noun for s in screens]
    out = []
    for index, screen in enumerate(screens):
        ordinal = None
        if nouns.count(screen.noun) > 1:
            ordinal = ("first", "second", "third", "fourth")[nouns[:index].count(screen.noun)]
        out.append(clause(screen, labels[index], ordinal))
    return " ".join(out)


_TRACKER_NAME = re.compile(
    r"^v(\d+)-(" + "|".join(PATTERNS) + r")-(" + "|".join(COLOUR) + r")-(\d+)x(\d+)\.png$")


def parse_tracker_file(name):
    """`tracker_file(...)`'s basename -> (pattern, colour, width, height), or None.

    What the tracker route and the writer use to draw a file from its name alone.
    """
    match = _TRACKER_NAME.match(name)
    if not match or int(match.group(1)) != TRACKER_VERSION:
        return None
    width, height = int(match.group(4)), int(match.group(5))
    if not (2 <= width <= 4096 and 2 <= height <= 4096):
        return None
    return match.group(2), match.group(3), width, height
