# Rendering from a script or a coding agent

Continuity can be driven without opening ComfyUI's page: from a shell, a CI
job, or a coding agent like Claude Code that you ask to "make a clip of a cat".
The pack ships a small command-line client, `cli/render.py`, that talks to a
running ComfyUI over HTTP. It needs only Python 3.9 or newer (no packages to
install) and runs on the ComfyUI machine or on any machine that can reach its
port.

## Quick start

From the pack's folder (`ComfyUI/custom_nodes/ComfyUI-Continuity`):

```
python3 cli/render.py families
python3 cli/render.py h3 "a cat stretches on a sunny windowsill"
```

The first command lists every model family, whether its files are installed,
and which turbo files a fast render would use. The second renders a six-second
MiniMax H3 clip, waits for it, downloads it to `./renders/` and prints the path
of the downloaded file.

The client talks to `http://127.0.0.1:8188` unless you set `--url` or the
`COMFY_URL` environment variable:

```
export COMFY_URL=http://192.168.1.20:8188
python3 cli/render.py krea2 "a tabby cat, studio portrait" --aspect 4:5
```

## What it decides for you

- **Weights.** The server picks each model file the way the node does: the
  files you last used on that machine, or the only file in a folder whose name
  fits. A file can be overridden with `--model SLOT=FILE`; `families` lists the
  slot names.
- **Speed.** A render is fast by default. The client turns on the family's turbo
  switch, the same one on the node's sampler row: a distilled checkpoint where
  the family has one, or a distillation LoRA from `models/loras`. For MiniMax H3
  it picks the FL2V or Ref2V distill to match the checkpoint the render
  actually uses. `--quality draft|medium|good` picks the step count.
  `--native` renders at the family's full step count instead, which for video
  can take many times longer.
- **Clip or picture.** Video families make a clip, and image families (Krea 2,
  Ideogram 4, Qwen, Flux 2 Klein) make a picture.

If the server can't tell which LoRA is the turbo one, because there is none or
there are several, it refuses the render and names the candidates. Pass one
with `--turbo-lora NAME`, or use `--native`. It never falls back to a slow render
without saying so.

## Attaching pictures

```
python3 cli/render.py h3 "@pic-1 turns and walks out of frame" --image cat.png
python3 cli/render.py h3 "@pic-1 and @pic-2 meet in a park" --image a.png:ref --image b.png:ref
```

`--image PATH[:AS]` is repeatable. A local file is uploaded to ComfyUI's input
folder. Any other name refers to a file already there, including an earlier
render such as `"H3_00012_.png [output]"`. Files are cited in the prompt in the
order given: pictures as `@pic-1`, `@pic-2`, videos as `@clip-1`, sounds as
`@snd-1`.

`AS` decides what a picture does in a clip. `start` makes it the opening frame
and `end` the closing frame. `ref` makes it a reference the model draws from, and
a scope such as `style` or `person` makes it a reference for that one thing.
Without `AS`, the first picture opens the shot.

## Other options

| Option | Meaning |
|---|---|
| `--seconds N` | clip length; the family's default otherwise |
| `--aspect 16:9` | shape; `9:16`, `1:1`, `4:5` and the others the node offers |
| `--edge N` | short edge in pixels; the family's native size otherwise |
| `--seed N` | random otherwise; printed either way |
| `--still` | a picture rather than a clip |
| `--out DIR` | where to download; `./renders` by default |
| `--no-wait` | queue it, print the prompt id, and return |

The client prints only the downloaded paths on stdout. Progress and errors go to
stderr, and the exit status is non-zero when a render can't be built or fails,
with the server's own explanation. A render queued this way is an ordinary
ComfyUI job: it shows in the queue, Cancel stops it, and the file lands in the
output folder like any other.

## For coding agents

Point your agent at this page, or at `python3 cli/render.py --help`, which says
the same thing more briefly. For Claude Code, a line in your project's
`CLAUDE.md` is enough:

```
To render with ComfyUI, use `python3 <path-to-pack>/cli/render.py` (see its --help).
Run `families` first; never write ComfyUI workflow JSON by hand.
```

## The HTTP API underneath

The client is a thin wrapper over two routes, which you can call from any
language.

`GET /continuity/render` returns what the machine can render: each family's id,
whether it is ready, the files it would use, and what a fast render would use.

`POST /continuity/render` takes JSON and queues one render:

```json
{"family": "h3", "prompt": "@pic-1 walks off", "pictures": [{"filename": "cat.png", "as": "start"}],
 "seconds": 6, "aspect": "16:9", "short_edge": 768, "seed": 7,
 "fast": true, "quality": "good", "turbo_lora": null, "still": false,
 "models": {"clip": "some_encoder.safetensors"}}
```

Only `family` and `prompt` are required. It answers
`{"prompt_id", "speed", "piece"}`, or `{"problem": "..."}` with status 400
when the render can't be built. Upload files first with ComfyUI's own
`POST /upload/image`. Poll `GET /history/<prompt_id>`, where the saved file
appears under the output key `mmc_video` (clips) or `mmc_image` (pictures), and
download it with `GET /view`. Send the body as `application/json`: the pack
refuses cross-site requests.
