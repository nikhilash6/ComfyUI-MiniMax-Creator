"""What screens add to a graph — and that without them, they add nothing.

    COMFYUI_PATH=~/ComfyUI <comfy-venv>/bin/python3 tests/test_screens_graph.py

Nothing is sampled. The claims pinned here are the ones a wrong graph would get
wrong silently: that the screen pass is the last thing before the save, after
the DLSS 5 refiner that would soften its text; that the tracker is on disk
before anything reads it; that a strip with a screen writes its takes off the
replaced reel; that a still's screen node sits between its decode and its save;
and that a piece without screens emits the graph it always did.

The trackers are drawn into a temporary temp folder, not the install's, and the
sweep of trackers older versions left in input/ runs against a temporary one.
Skips itself with a message if ComfyUI cannot be imported.
"""

import asyncio
import importlib
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE = os.path.basename(ROOT)
COMFY = os.environ.get("COMFYUI_PATH", os.path.expanduser("~/ComfyUI"))
BASE = os.environ.get("COMFYUI_BASE", COMFY)


def _boot():
    sys.path.insert(0, COMFY)
    sys.argv = ["main.py", "--base-directory", BASE]
    import nodes
    import server

    loop = asyncio.new_event_loop()
    try:
        from app.assets.manager import default_asset_manager
        server.PromptServer(loop, default_asset_manager())
    except (ImportError, TypeError):
        server.PromptServer(loop)
    asyncio.set_event_loop(loop)
    loop.run_until_complete(nodes.init_extra_nodes(init_custom_nodes=False))

    sys.path.insert(0, os.path.dirname(ROOT))
    return nodes


try:
    comfy_nodes = _boot()
except Exception as exc:  # noqa: BLE001
    print(f"skipped: ComfyUI not importable ({type(exc).__name__}: {exc})")
    sys.exit(0)

import folder_paths  # noqa: E402

package = importlib.import_module(PACKAGE)
cn = importlib.import_module(f"{PACKAGE}.creator.creator_node")
ps = importlib.import_module(f"{PACKAGE}.creator.prestage")
screen_node = importlib.import_module(f"{PACKAGE}.creator.screens.node")
screen_spec = importlib.import_module(f"{PACKAGE}.creator.screens.spec")
settings = importlib.import_module(f"{PACKAGE}.creator.settings")

from harness import check, passed  # noqa: E402

passed("the screen pass is last before the save, its trackers are drawn first, "
       "and a piece without screens is the graph it always was")

PASS, STILL = screen_node.PASS_NODE, screen_node.STILL_NODE
NODE_ID = "7"
H3_MODELS = {"fl2va": "h3/fl2va.safetensors", "ref2va": "h3/ref2va.safetensors",
             "clip": "h3/text_encoder.safetensors", "vae": "h3/video_vae.safetensors",
             "audio_vae": "h3/audio_vae.safetensors"}
KLEIN_MODELS = {"flux2klein": {"model": "klein.safetensors", "clip": "qwen3.safetensors",
                               "vae": "flux2-vae.safetensors"}}
PHONE = {"handle": "phone-1", "device": "phone",
         "content": {"filename": "guide.png", "kind": "image"}}
NEURAL = {"on": True, "profile": "standard", "detail": 1.0, "colour": 1.0,
          "intensity": 1.0, "precision": "fast"}

# The graph reads this machine's tracker off the settings file; pinned here so
# the suite does not depend on what the settings page last said.
settings.screen_tracker = lambda: dict(screen_spec.DEFAULT_STYLE)
inputs = tempfile.mkdtemp(prefix="continuity-screens-")
folder_paths.set_input_directory(inputs)
temp = tempfile.mkdtemp(prefix="continuity-screens-temp-")
folder_paths.set_temp_directory(temp)


def with_id(node_class, unique_id, run):
    from comfy_api.latest import io as comfy_io

    previous = node_class.hidden
    node_class.hidden = comfy_io.HiddenHolder(
        unique_id=unique_id, prompt=None, extra_pnginfo=None, dynprompt=None,
        auth_token_comfy_org=None, api_key_comfy_org=None)
    try:
        return run()
    finally:
        node_class.hidden = previous


def piece(segments, **extra):
    return json.dumps({"version": 2, "prompt": "", "models": H3_MODELS, "aspect": "3:4",
                       "short_edge": 768, "segments": segments, **extra})


def video(data):
    return with_id(cn.MiniMaxH3Creator, NODE_ID,
                   lambda: cn.MiniMaxH3Creator.execute(
                       creator_data=data, seed=100, steps=8, cfg=1.0,
                       sampler_name="res_multistep", scheduler="simple")).expand


def still(data):
    return with_id(ps.MiniMaxH3PreStage, NODE_ID,
                   lambda: ps.MiniMaxH3PreStage.execute(
                       prestage_data=json.dumps(data), seed=100, steps=4, cfg=1.0,
                       sampler_name="euler", scheduler="simple")).expand


def by_class(graph):
    out = {}
    for node_id, node in graph.items():
        out.setdefault(node["class_type"], []).append((node_id, node["inputs"]))
    return out


SHOT = {"prompt": "A hand holds a phone in a gallery; it shows @phone-1.", "duration_s": 5}
TRACKER = os.path.join(temp, "continuity_trackers", "v2-none-green-1080x2340.png")

# --- a video ---------------------------------------------------------------------

# What an older version left behind: its trackers, beside a file of the user's
# that happens to sit in the same folder.
old = os.path.join(inputs, "continuity_trackers")
os.makedirs(old)
for name in ("v1-grid-white-1080x2340.png", "v2-plus-green-1920x1080.png", "mine.png"):
    open(os.path.join(old, name), "wb").close()

BARE = {**SHOT, "prompt": "A hand holds a phone in a gallery."}
check("a piece with no screens emits no pass", PASS in by_class(video(piece([BARE]))), False)
check("...and draws no tracker", os.path.exists(TRACKER), False)

kinds = by_class(video(piece([{**SHOT, "screens": [PHONE]}], neural=NEURAL)))
check("one pass for the reel", len(kinds.get(PASS, [])), 1)
pass_id, pass_inputs = kinds[PASS][0]
refine_id, _ = kinds["ContinuityNeuralPass"][0]
check("it reads the refined reel: the refiner would soften the screen's text",
      pass_inputs["reel"], [refine_id, 0])
check("the file is written from what it hands back",
      kinds["MiniMaxH3Save"][0][1]["reel"], [pass_id, 0])
data = json.loads(pass_inputs["screens_data"])
check("it is told which part has which screen, with this machine's tracker",
      {k: [(s["filename"], s["pattern"], s["colour"]) for s in v] for k, v in data.items()},
      {"0": [("guide.png", "none", "green")]})
check("a warning lands on the node the user is looking at", pass_inputs["report_to"], NODE_ID)
check("the tracker is on disk before the segment that reads it runs", os.path.isfile(TRACKER), True)

two = [{**SHOT, "screens": [PHONE]}, {"prompt": "the gallery at night", "duration_s": 5}]
kinds = by_class(video(piece(two)))
check("on a strip only the part with a screen is replaced",
      list(json.loads(kinds[PASS][0][1]["screens_data"])), ["0"])
check("a strip with a screen writes its takes off the replaced reel, not per pass",
      "ContinuityTake" in kinds, False)
check("...where the same strip without one writes them per pass",
      "ContinuityTake" in by_class(video(piece([BARE, two[1]]))), True)

# --- a still ----------------------------------------------------------------------

klein = {"version": 1, "arch": "flux2klein", "prompt": "a hand holding a phone",
         "aspect": "3:4", "short_edge": 1024, "refs": [], "loras": [], "models": KLEIN_MODELS}
check("a still with no screens emits no screen node", STILL in by_class(still(klein)), False)
kinds = by_class(still({**klein, "prompt": "a hand holding a phone showing @phone-1",
                         "screens": [PHONE]}))
check("one screen node on a still", len(kinds.get(STILL, [])), 1)
still_id, still_inputs = kinds[STILL][0]
check("the save writes the replaced picture", kinds["MiniMaxH3SaveImage"][0][1]["images"],
      [still_id, 0])
check("the screen node reads the decode",
      any(node_id == still_inputs["image"][0] for node_id, _ in kinds.get("VAEDecode", [])), True)
check("the tracker is loaded as the still's reference",
      any(inputs_.get("image") == "continuity_trackers/v2-none-green-1080x2340.png [temp]"
          for _, inputs_ in kinds.get("LoadImage", [])), True)

check("the trackers an older version put in input/ are swept, and nothing else is",
      sorted(os.listdir(old)), ["mine.png"])

# --- the debug files -----------------------------------------------------------------

settings.screen_debug = lambda: True
kinds = by_class(video(piece([{**SHOT, "screens": [PHONE]}])))
saves = kinds["MiniMaxH3Save"]
prefixes = sorted(inputs_["filename_prefix"].rsplit("-", 1)[-1] if inputs_["filename_prefix"].endswith(("-raw", "-screens"))
                  else "final" for _, inputs_ in saves)
check("with the debug switch on, the raw render and the pass's view are saved too",
      prefixes, ["final", "raw", "screens"])
pass_id = kinds[PASS][0][0]
check("the view is the pass's second output",
      any(inputs_["reel"] == [pass_id, 1] for _, inputs_ in saves), True)
settings.screen_debug = lambda: False
