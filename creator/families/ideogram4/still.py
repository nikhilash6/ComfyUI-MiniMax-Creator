"""Ideogram 4.0's own half of an image render: the preset table and the
dual-model sampler branch.

The shared flow is `compile_image.compile_prestage` and `render_image.emit`,
which take this module as the `family`. What stays here is what only Ideogram
knows: its knobs are the official preset table rather than a steps widget
alone (mu and std shape the resolution-shifted schedule, so they belong to the
preset, not to the user), its unconditional branch is a separate checkpoint
behind `DualModelGuider`, and its schedule is `Ideogram4Scheduler`'s — which is
why this branch samples through `SamplerCustomAdvanced` rather than `KSampler`.

The sampler row is the other thing only Ideogram knows, and none of it is the
template's: the conditional branch runs behind `ModelSamplingAuraFlow` at shift
5 with res_2m, and the guidance tail starts at a fixed percent — see the
constants below for what the stock row did instead.

Two more things are Ideogram's alone. Its prompt is a JSON caption, not prose
(`format_prompt`), and a LoRA over it is two patches — one per checkpoint, each
at its own weight (`emit_graph`).
"""

import json
import math
import sys
from statistics import NormalDist

from . import magic

ARCH = "ideogram4"

# Which weights fields this architecture has. No distilled checkpoint — the
# speed axis is the preset table; the second file is the unconditional branch.
FIELDS = ("model", "uncond_model", "clip", "vae")

# Which VAE the `vae` field has to hold, checked off the file's header before
# the render is queued — see `vaekind`.
VAE_KIND = "flux2"

# What CLIPLoader calls the Qwen3-VL-8B encoder.
CLIP_TYPE = "ideogram4"

# The model reads no reference conditioning at all; a render that silently
# ignored attached images is the failure this package exists to avoid, so the
# shared compile refuses them with this.
TAKES_REFS = False
REFS_REFUSAL = (
    "Ideogram 4.0 has no local reference conditioning — switch the "
    "model pill to Krea 2 or Qwen Image Edit, or clear the style references"
)

# Ideogram's official preset table (V4_QUALITY_48 / V4_DEFAULT_20 /
# V4_TURBO_12): mu and std shape the resolution-shifted schedule, so they
# belong to the preset, not to the user. The one departure is Turbo's step
# count — 18 rather than the template's 12, on the same mu/std, which is the
# row the model renders cleanly at (measured 2026-09-20 against the stock
# template, which on the same weights and prompt drew the model's built-in
# "blocked by safety filter" card). `polish` is the tail each preset ends on
# at the lower guidance weight, in steps.
IDEOGRAM_QUALITIES = {
    "quality": {"steps": 48, "mu": 0.0, "std": 1.5, "polish": 3},
    "default": {"steps": 20, "mu": 0.0, "std": 1.75, "polish": 2},
    "turbo": {"steps": 18, "mu": 0.5, "std": 1.75, "polish": 1},
}
DEFAULT_IDEOGRAM_QUALITY = "turbo"
# Guidance is 7 on every preset — the preset moves the step count and the
# schedule, never the weight — and drops to 3 for the polish tail so the fine
# steps stop over-sharpening. The 7 is the node's cfg widget; the 3 is constant.
IDEOGRAM_CFG = 7.0
IDEOGRAM_CFG_POLISH = 3.0
# Where the tail starts, as the percent `CFGOverride` takes. Fixed rather than
# resolved against the schedule: the conditional branch samples behind
# `ModelSamplingAuraFlow` at `IDEOGRAM_SHIFT`, and the override converts its
# percent to a sigma through that same patch, so step arithmetic done on the
# raw `sigmas` (shift 1) would not say where a step falls.
IDEOGRAM_POLISH_START = 0.7
# The conditional branch runs shifted; the unconditional checkpoint does not.
# Together with res_2m this is the sampler row the model is clean on — at
# shift 1 with euler, the same weights and prompt come back as the model's
# refusal card.
IDEOGRAM_SHIFT = 5.0
IDEOGRAM_SAMPLER = "res_2m"

# The other speed axis, and the one the preset table cannot reach: a distillation
# LoRA over the same checkpoint, which takes Ideogram down to a handful of steps
# at cfg 1. There is no distilled Ideogram checkpoint to swap to — the presets
# *are* the official fast path — so unlike Krea's pill this one is a LoRA or it
# is nothing, and the compile says so rather than sampling 4 steps of an
# undistilled model at cfg 1 and calling it turbo.
#
# At cfg 1 two things stop being worth their weight: the unconditional branch is
# never evaluated, so its 9.3B file is not loaded at all, and the polish tail has
# no guidance left to drop. The schedule stays the Turbo preset's, which is the
# one shaped for a short run.
# The stops and the row are TurboTime's author's (ostris; the nomadoor guide
# carries the workflow): "a 2 to 8 step LoRA, but very low step counts clearly
# start to break the image" — so 8 is the stop it opens on — at strength 0.6,
# on euler, cfg 1, no unconditional model. Measured 2026-09-20: at 1.0 under
# a content LoRA at 0.9 the frame comes back oversaturated and plastic.
TURBO_STEPS = {"draft": 2, "medium": 4, "good": 8}
DEFAULT_TURBO_QUALITY = "good"
DEFAULT_TURBO_STRENGTH = 0.6
TURBO_QUALITY = "turbo"
TURBO_ROW = {"cfg": 1.0, "sampler_name": "euler"}
TURBO_NEEDS_LORA = (
    "Ideogram 4.0 has no distilled checkpoint — its turbo pill is a "
    "distillation LoRA over the ordinary one. Pick the LoRA, or switch the "
    "quality pill to the Turbo preset instead"
)

# `Ideogram4Scheduler`'s logSNR clamp, mirrored from core's `nodes_ideogram4`
# so the schedule can be reasoned about (and tested) without a ComfyUI to
# import.
_LOGSNR_MIN = -15.0
_LOGSNR_MAX = 18.0
_NORMAL = NormalDist()


def sigmas(steps, width, height, mu, std):
    """The descending sigmas `Ideogram4Scheduler` will hand the sampler.

    `core.nodes_ideogram4.ideogram4_sigmas` in plain Python: a logit-normal
    schedule whose mean carries a resolution term, which is why the same preset
    lands its steps somewhere else on a 2K canvas than on a 1K one.
    """
    mean = mu + 0.5 * math.log((width * height) / (512 * 512))
    t_min = 1.0 / (1.0 + math.exp(0.5 * _LOGSNR_MAX))
    t_max = 1.0 / (1.0 + math.exp(0.5 * _LOGSNR_MIN))
    out = []
    for i in range(steps + 1):
        u = (steps - i) / steps
        # The quantile blows up at the ends; the clamp below is where both
        # infinities land anyway, so they are taken there directly.
        if u <= 0.0:
            t = t_max
        elif u >= 1.0:
            t = t_min
        else:
            t = 1.0 / (1.0 + math.exp(mean + std * _NORMAL.inv_cdf(u)))
            t = min(max(t, t_min), t_max)
        out.append(1.0 - t)
    out[-1] = 0.0                       # core forces the last sigma to zero
    return out


def format_prompt(prompt):
    """The prose as the JSON caption the model was trained on.

    Ideogram's authors are blunt about it: a plain-text prompt "will not work
    and will likely trigger a safety warning" — the grey "blocked by safety
    filter" card. The schema's floor is one `compositional_deconstruction`
    with a `background` and an `elements` list; `high_level_description` is
    the short sentence a magic-prompt would open with. Without a language model
    in the loop the prose is all three, which is the schema-valid caption the
    model reads rather than the plain line it refuses.

    A prompt that already *is* the schema — pasted from a workflow, or written
    by the chat's magic prompt (`magic.py`) — keeps every word as written and
    is only put in the key order the model was trained on, minified
    (`magic.tidy`).
    """
    text = prompt.strip()
    if text.startswith("{"):
        tidied = magic.tidy(text)
        if tidied is not None:
            return tidied
    return json.dumps({
        "high_level_description": text,
        "compositional_deconstruction": {
            "background": text,
            "elements": [{"type": "obj", "desc": text}],
        },
    }, ensure_ascii=False)


def plan(data):
    """The blob's arch-specific decisions -> (checkpoint field, schedule).

    The schedule block is the whole preset, step count included. Steps do not
    ride the sampler row here the way every other family's do: the preset pill
    is what the user picks, and a row's steps are a copy of it that can go
    stale — a blob written under an older preset table queued 12 steps under a
    pill reading "turbo · 18", and nothing on screen said so. Derived from the
    preset (or the turbo pill's quality stop), the pill cannot lie. What is
    left in the block is the shape the scheduler is given, the tail the
    guidance drops over, and whether the unconditional branch is worth
    loading. There is one checkpoint field either way — the turbo pill here is
    a LoRA, not a second file.
    """
    from ...compile import CompileError

    from ...compile_image import turbo_block

    turbo = turbo_block(data, ARCH)
    if turbo.get("on"):
        if not turbo.get("lora"):
            raise CompileError(TURBO_NEEDS_LORA)
        preset = IDEOGRAM_QUALITIES[TURBO_QUALITY]
        steps = TURBO_STEPS.get(turbo.get("quality"), TURBO_STEPS[DEFAULT_TURBO_QUALITY])
        return "model", {"steps": steps, "mu": preset["mu"], "std": preset["std"],
                         "polish": 0, "uncond": False}

    quality = data.get("quality", DEFAULT_IDEOGRAM_QUALITY)
    if quality not in IDEOGRAM_QUALITIES:
        raise CompileError(f"unknown Ideogram quality preset {quality!r}")
    preset = IDEOGRAM_QUALITIES[quality]
    return "model", {"steps": preset["steps"], "mu": preset["mu"], "std": preset["std"],
                     "polish": preset["polish"], "uncond": True}


def require_support():
    """Refuse a core that does not know Ideogram 4 yet — see krea2's twin."""
    import nodes

    if "Ideogram4Scheduler" not in nodes.NODE_CLASS_MAPPINGS:
        raise ValueError(
            "This ComfyUI does not know Ideogram 4 yet (no Ideogram4Scheduler "
            "node). Update ComfyUI and restart."
        )


def emit_graph(graph, payload, sampling, weights, clip, vae, model, unique_id,
               filename_prefix):
    """The sampler branch over the shared prologue's loaders."""
    from ... import render_image
    from ...compile import CompileError

    # Guidance is the whole point of the second checkpoint: at cfg 1 the
    # unconditional branch is never evaluated and the model draws unguided —
    # dark, soft and off the prompt. That is the distilled row's number, and it
    # is only right under the turbo LoRA (which `plan` marks by dropping the
    # branch). Anywhere else it is a stale row, and a stale row is refused by
    # name rather than sampled into mush.
    if payload.schedule.get("uncond") and sampling.cfg <= 1.0:
        raise CompileError(
            f"Ideogram 4.0 at cfg {sampling.cfg:g} samples without guidance — that "
            f"is the turbo LoRA's row. Set cfg back to {IDEOGRAM_CFG:g}, or throw "
            "the turbo pill.")

    positive = graph.node("CLIPTextEncode", clip=clip, text=payload.prompt).out(0)
    negative = graph.node("ConditioningZeroOut", conditioning=positive).out(0)

    # The shift and the polish tail both wrap the conditional model, after the
    # LoRAs — sampling patches, not weight patches. The unconditional branch
    # below is loaded bare. Neither is applied under the turbo LoRA: the
    # distilled run has no guidance tail for the shift to place, and its
    # author's row samples the model unshifted.
    if payload.schedule.get("polish", 0) > 0:
        model = graph.node("ModelSamplingAuraFlow", model=model,
                           shift=IDEOGRAM_SHIFT).out(0)
        model = graph.node("CFGOverride", model=model, cfg=IDEOGRAM_CFG_POLISH,
                           start_percent=IDEOGRAM_POLISH_START,
                           end_percent=1.0).out(0)

    # Without the unconditional file picked, `DualModelGuider` degrades to
    # ordinary CFG on the one model, which the node itself documents — so that
    # file is optional here too.
    guider_inputs = {"model": model, "positive": positive, "negative": negative,
                     "cfg": sampling.cfg}
    if payload.schedule.get("uncond") and weights.get("uncond_model"):
        # A LoRA over Ideogram is two patches, one per checkpoint: the shared
        # prologue put it on the conditional branch at `strength`, and here it
        # goes on the unconditional one at its own `uncond` weight — the same
        # file, usually lighter. At zero it is left off this branch.
        uncond = render_image.emit_unet(graph, weights, "uncond_model")
        for entry in payload.loras:
            if entry.get("uncond", entry["strength"]) == 0.0:
                continue
            uncond = graph.node("LoraLoaderModelOnly", model=uncond,
                                lora_name=entry["name"],
                                strength_model=entry.get("uncond", entry["strength"])).out(0)
        guider_inputs["model_negative"] = uncond
    guider = graph.node("DualModelGuider", **guider_inputs).out(0)

    schedule = graph.node("Ideogram4Scheduler", steps=payload.schedule["steps"],
                          width=payload.width, height=payload.height,
                          mu=payload.schedule["mu"],
                          std=payload.schedule["std"]).out(0)
    latent, denoise = render_image.emit_latent(graph, payload, vae,
                                               "EmptyFlux2LatentImage")
    if denoise < 1.0:
        # img2img on a custom schedule: keep the tail of the sigmas and let the
        # noise node start the latent at the truncated schedule's first sigma —
        # the same statement KSampler's denoise makes, said in sigmas.
        schedule = graph.node("SplitSigmasDenoise", sigmas=schedule,
                              denoise=denoise).out(1)

    sampled = graph.node(
        "SamplerCustomAdvanced",
        noise=graph.node("RandomNoise", noise_seed=sampling.seed).out(0),
        guider=guider,
        sampler=graph.node("KSamplerSelect", sampler_name=sampling.sampler_name).out(0),
        sigmas=schedule, latent_image=latent,
    )
    render_image.emit_tail(graph, sampled.out(0), vae, unique_id, filename_prefix,
                           request=payload.neural, screens=payload.screens)


def compile_still(data, image_size_lookup=None):
    """The uniform still surface — see `families/registry.py`. The flow is the
    shared `compile_image.compile_prestage`, handed this module as the family."""
    from ... import compile_image

    return compile_image.compile_prestage(data, sys.modules[__name__],
                                          image_size_lookup)


def emit_still(data, plan, sampling, unique_id):
    """The uniform still surface over the shared `render_image.emit`."""
    from ... import outputs, render_image, settings
    from . import declare

    weights = render_image.ImageWeights.from_blob(data, sys.modules[__name__])
    return render_image.emit(plan, weights, sampling, unique_id,
                             sys.modules[__name__],
                             filename_prefix=outputs.image(
                                 data, settings.image_prefix(declare.ID)))
