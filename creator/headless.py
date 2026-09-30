"""A render asked for without a browser: what `/continuity/render` adds to the chat's builder.

The chat room already turns "this prompt, these pictures, this family" into a
validated one-node prompt (`routes/chat._build`), and that is the whole of what
a script needs too — the same patch, the same weight picks, the same dry run.
What the room gets from the canvas and a script has nowhere to get from is the
**turbo switch**. On the node it is thrown in the browser (`web/creator/turbo.js`,
the pre-stage's `throwTurbo`): a distillation LoRA put on the stack, or the
distilled checkpoint swapped in, and the sampler row moved to the few-step
numbers the family declares. A script asking for "fast" has to get the same
blob, so this module throws it the same way, off the same declarations
(`capabilities.turbo` in each family's manifest).

**Which LoRA is the one open question, and it is answered like a weight slot.**
Nothing on a machine records which file in `models/loras` is a family's
distillation — on the node the person picks it once, per piece. So the rule is
`chat.guess_weights`' rule: a file is taken when exactly one name fits the
family's `hints` (and, on a family that routes between checkpoints, that
checkpoint's own — H3's FL2V and Ref2V distills are separate files and each is
wrong on the other's weights). A tie is broken by the step count a name
carries against the quality asked for, since the published H3 distills differ
by exactly that; anything still ambiguous is refused with the candidates named,
because a coin toss between two distillations is a question for the person.

**No quiet slow path.** A family whose switch cannot be thrown here — no file
fits — is refused rather than rendered at its native row. A native H3 render is
an hour where a turbo one is minutes, and the caller asked for the fast one;
rendering the other without saying so is the fallback this pack does not do.
The caller says `fast: false` to mean native.

**A merged checkpoint is said, not guessed.** Some checkpoints carry the
distillation in their own weights (a hybrid H3 merge, say) and want the step
drop with no LoRA at all — the node's "no LoRA · merged checkpoint" choice.
Nothing in a filename says so reliably, so the caller says `merged: true`, and
the switch then writes only what the node's does in that mode: the step table
and the row, with the flow shifts left at the checkpoint's own.

Pure: no ComfyUI, no disk. The route hands in the manifest, the picks and the
LoRA names.
"""

import re

from .chat import TURBO_SLOT

# What makes a LoRA's name read as a distillation at all, before the family's
# own hints narrow it: the words the published files carry.
TURBO_WORDS = re.compile(
    r"turbo|distill|lightning|lightx2v|pdd|acc[-_]?\d+step|\d+[-_]?steps?(?![a-z])",
    re.IGNORECASE)
STEP_COUNT = re.compile(r"(\d+)[-_]?steps?(?![a-z])", re.IGNORECASE)


class HeadlessError(ValueError):
    """A request this route cannot build, as the sentence the caller reads."""


def turbo_of(family):
    """The family's turbo declaration, or None where it has no switch."""
    return (family.get("capabilities") or {}).get("turbo") or None


def _quality(turbo, quality):
    quality = quality or turbo.get("default_quality")
    if quality not in turbo["steps"]:
        raise HeadlessError(f"quality must be one of {', '.join(turbo['steps'])}; "
                            f"yours was {quality!r}.")
    return quality


def candidates(turbo, names, checkpoint=None):
    """Every LoRA name that could be this family's distillation for `checkpoint`."""
    hints = turbo.get("hints") or {}
    family_re = hints.get("family")
    own_re = (hints.get("checkpoints") or {}).get(checkpoint)
    return [name for name in names
            if TURBO_WORDS.search(name)
            and (not family_re or re.search(family_re, name, re.IGNORECASE))
            and (not own_re or re.search(own_re, name, re.IGNORECASE))]


def pick_lora(turbo, names, steps, checkpoint=None, label=""):
    """The one distillation that fits, or `HeadlessError` naming the candidates."""
    found = candidates(turbo, names, checkpoint)
    if len(found) > 1:
        # The step count a file was distilled for, where its name says one.
        matching = [name for name in found
                    if any(int(n) == steps for n in STEP_COUNT.findall(name))]
        if len(matching) == 1:
            found = matching
    if len(found) == 1:
        return found[0]
    what = f"{label} ({checkpoint})" if checkpoint else label
    if not found:
        raise HeadlessError(
            f"no turbo LoRA for {what} is in models/loras, so this cannot render "
            f"fast. Name one with turbo_lora, say merged: true if the checkpoint "
            f"has the distillation merged in, or ask for a native render "
            f"(fast: false) — at the family's full step count.")
    raise HeadlessError(
        f"more than one LoRA could be the turbo for {what}: "
        f"{', '.join(found)}. Name the one to use with turbo_lora, or say "
        f"merged: true if the checkpoint has the distillation merged in.")


def preset(turbo, name):
    """What a distill file engages at — `state.turboPreset`, said here for a
    render with no browser. The first preset whose `match` is in the name, else
    the family's own row and the checkpoints' own schedule."""
    for entry in turbo.get("presets") or []:
        if re.search(entry["match"], name or "", re.IGNORECASE):
            return {"strength": entry["strength"],
                    "shift_video": entry["shift_video"], "shift_audio": entry["shift_audio"],
                    "row": entry.get("row") or turbo["row"],
                    "steps": entry.get("steps") or turbo["steps"]}
    reset = turbo.get("reset") or {}
    shifts = {key: reset[key] for key in ("shift_video", "shift_audio") if key in reset}
    return {"strength": turbo.get("default_strength", 1.0), **shifts,
            "row": turbo["row"], "steps": turbo["steps"]}


def throw_video(piece, family, checkpoints, names, lora=None, quality=None, merged=False):
    """Put the video family's turbo switch on `piece`, in place -> what it did.

    `checkpoints` are the routed checkpoints the dry run said this piece samples
    on. Each gets its own distillation, claimed for it alone (`modes`), so the
    stack never patches an FL2V distill onto Ref2V weights; the row is the
    first one's, and on a piece of one shot there is only one.

    `merged` is the checkpoint carrying its own distillation: no LoRA, the
    family's step table and row, the shifts untouched — `turbo.js`' `throwOn`
    with no file.
    """
    turbo = turbo_of(family)
    if turbo is None:
        return None
    quality = _quality(turbo, quality)
    if merged:
        if lora:
            raise HeadlessError("turbo_lora and merged: true contradict each other — a "
                                "merged checkpoint takes no turbo LoRA.")
        steps = turbo["steps"][quality]
        piece["turbo"] = {"lora": None, "on": True, "quality": quality, "merged": True}
        piece["sampling"] = {**(piece.get("sampling") or {}), "steps": steps, **turbo["row"]}
        return {"quality": quality, "steps": steps, "loras": [], "merged": True}
    routed = [c for c in checkpoints if c] or [None]
    steps_wanted = turbo["steps"][quality]
    files = {c: lora or pick_lora(turbo, names, steps_wanted, c, family["label"])
             for c in routed}
    first = files[routed[0]]
    engaged = preset(turbo, first)
    piece["loras"] = [entry for entry in piece.get("loras") or []
                      if entry.get("name") not in files.values()]
    for checkpoint, name in files.items():
        piece["loras"].append({"name": name, "strength": preset(turbo, name)["strength"],
                               "enabled": True, "triggers": [],
                               "modes": [checkpoint] if checkpoint else []})
    piece["turbo"] = {"lora": first, "on": True, "quality": quality}
    steps = engaged["steps"].get(quality, steps_wanted)
    piece["sampling"] = {**(piece.get("sampling") or {}), "steps": steps,
                         **engaged["row"],
                         **{key: engaged[key] for key in ("shift_video", "shift_audio")
                            if key in engaged}}
    return {"quality": quality, "steps": steps, "loras": sorted(set(files.values()))}


def throw_still(piece, family, picks, names, lora=None, quality=None):
    """Put an image family's turbo switch on a pre-stage `piece`, in place.

    The distilled checkpoint where the family has one and this machine has it
    picked, which is what the pill throws first; a LoRA otherwise, on the
    families whose switch is one.
    """
    turbo = turbo_of(family)
    if turbo is None:
        return None
    arch = piece["arch"]
    quality = _quality(turbo, quality)
    steps = turbo["steps"][quality]
    block = {"on": True, "quality": quality}
    said = {"quality": quality, "steps": steps}
    if not lora and turbo.get("checkpoint") and picks.get(TURBO_SLOT):
        block["lora"] = None
        said["checkpoint"] = picks[TURBO_SLOT]
    elif turbo.get("lora"):
        name = lora or pick_lora(turbo, names, steps, label=family["label"])
        block["lora"] = name
        piece["loras"] = [entry for entry in piece.get("loras") or [] if entry.get("name") != name]
        piece["loras"].append({"name": name, "strength": turbo.get("default_strength", 1.0),
                               "enabled": True})
        said["loras"] = [name]
    else:
        raise HeadlessError(
            f"{family['label']}'s turbo is its distilled checkpoint, and none is "
            f"picked on this machine — pick one for '{TURBO_SLOT}' (models: "
            f"{{\"{TURBO_SLOT}\": ...}}), or ask for a native render (fast: false).")
    piece["turbo"] = {**(piece.get("turbo") or {}), arch: block}
    piece["sampling"] = {**(piece.get("sampling") or {}), "steps": steps, **turbo["row"]}
    return said
