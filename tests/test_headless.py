"""The turbo switch, thrown with no browser: `creator/headless.py`.

    python3 tests/test_headless.py

Runs standalone — no torch, no ComfyUI. The LoRA names are a real machine's
(the lab's `models/loras`, 2026-09-28), because what this module has to get
right is exactly the mess real names are: two families' distills side by side,
H3's split by checkpoint, and two Ref2V files that differ only by step count.
The row and strengths it writes are held against the family's own declaration,
which is what the node's switch (`web/creator/turbo.js`) writes from too.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import layout  # noqa: E402
from harness import FAILURES, check, passed  # noqa: E402

pkg = layout.load("prompting", "chat", "manifest", "headless")
headless, manifest = pkg.headless, pkg.manifest

LAB = [
    "ideogram_4_turbotime_v1.safetensors",
    "minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors",
    "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
    "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
    "styles/watercolor_h3.safetensors",
    "anna_person_v2.safetensors",
]
FL2V, REF4, REF8 = LAB[1], LAB[2], LAB[3]


def refuses(label, call, *fragments):
    try:
        call()
    except headless.HeadlessError as problem:
        for fragment in fragments:
            if fragment not in str(problem):
                FAILURES.append(f"{label}: {str(problem)!r} does not mention {fragment!r}")
    else:
        FAILURES.append(f"{label}: did not refuse")


h3 = manifest.describe("h3")
turbo = headless.turbo_of(h3)

# ---- which file -------------------------------------------------------------

check("an FL2V render takes the FL2V distill and nothing of Ref2V's",
      headless.candidates(turbo, LAB, "fl2va"), [FL2V])
check("a Ref2V render sees both Ref2V distills and no other family's",
      headless.candidates(turbo, LAB, "ref2va"), [REF4, REF8])
check("the tie is broken by the step count the quality asks for",
      (headless.pick_lora(turbo, LAB, 8, "ref2va"), headless.pick_lora(turbo, LAB, 4, "ref2va")),
      (REF8, REF4))
refuses("and a tie the step count cannot break is a question, with the candidates",
        lambda: headless.pick_lora(turbo, LAB, 6, "ref2va", "MiniMax H3"), REF4, REF8, "turbo_lora")
refuses("no distill at all refuses rather than rendering slow",
        lambda: headless.pick_lora(turbo, LAB[4:], 6, "fl2va", "MiniMax H3"),
        "no turbo LoRA", "fast: false")
check("Ideogram finds its own and not H3's",
      headless.candidates(headless.turbo_of(manifest.describe("ideogram4")), LAB), [LAB[0]])

# ---- the video switch -------------------------------------------------------

piece = {"family": "h3", "loras": [{"name": "anna_person_v2.safetensors", "strength": 0.8}]}
said = headless.throw_video(piece, h3, ["fl2va"], LAB, quality="good")
lightx2v = next(p for p in turbo["presets"] if p["match"] == "lightx2v")
check("the switch says what it threw", said, {"quality": "good", "steps": 8, "loras": [FL2V]})
check("the distill joins the stack after what was on it, claimed for its checkpoint alone",
      [(e["name"], e.get("modes")) for e in piece["loras"]],
      [("anna_person_v2.safetensors", None), (FL2V, ["fl2va"])])
check("at the strength its preset names", piece["loras"][1]["strength"], lightx2v["strength"])
check("the row is the family's turbo row with the preset's shifts",
      piece["sampling"], {"steps": 8, **turbo["row"],
                          "shift_video": lightx2v["shift_video"],
                          "shift_audio": lightx2v["shift_audio"]})
check("and the switch's memory is the node's shape", piece["turbo"],
      {"lora": FL2V, "on": True, "quality": "good"})

piece = {"family": "h3"}
headless.throw_video(piece, h3, ["fl2va", "ref2va"], LAB, quality="good")
check("a piece routing to both checkpoints carries a distill for each",
      [(e["name"], e["modes"]) for e in piece["loras"]], [(FL2V, ["fl2va"]), (REF8, ["ref2va"])])

piece = {"family": "h3"}
headless.throw_video(piece, h3, ["fl2va"], LAB, lora="my_merged_turbo.safetensors")
check("a named file is taken as named, at the family's default quality",
      (piece["loras"][0]["name"], piece["turbo"]["quality"]),
      ("my_merged_turbo.safetensors", turbo["default_quality"]))
refuses("an unknown quality is refused by name",
        lambda: headless.throw_video({}, h3, ["fl2va"], LAB, quality="best"), "draft", "good")
check("a family with no switch throws nothing",
      headless.throw_video({}, {"label": "X", "capabilities": {}}, [], LAB), None)

# ---- the still switch -------------------------------------------------------

krea = manifest.describe("krea2")
kturbo = headless.turbo_of(krea)
piece = {"arch": "krea2", "loras": []}
said = headless.throw_still(piece, krea, {"turbo_model": "krea2_turbo.safetensors"}, LAB)
check("Krea with its distilled checkpoint picked throws the checkpoint, no LoRA",
      (said.get("checkpoint"), piece["loras"], piece["turbo"]["krea2"]["lora"]),
      ("krea2_turbo.safetensors", [], None))
check("at the quality stop's steps on the family's turbo row", piece["sampling"],
      {"steps": kturbo["steps"][kturbo["default_quality"]], **kturbo["row"]})

ideo = manifest.describe("ideogram4")
iturbo = headless.turbo_of(ideo)
piece = {"arch": "ideogram4", "loras": []}
headless.throw_still(piece, ideo, {}, LAB)
check("Ideogram's switch is its distill LoRA at the declared strength",
      piece["loras"], [{"name": LAB[0], "strength": iturbo["default_strength"], "enabled": True}])

klein = manifest.describe("flux2klein")
refuses("a checkpoint-only switch with no checkpoint picked says which slot",
        lambda: headless.throw_still({"arch": "flux2klein"}, klein, {}, LAB), "turbo_model", "fast: false")

passed("the turbo switch is thrown headless as the node throws it")
