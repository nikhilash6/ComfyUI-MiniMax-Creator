---
name: continuity-render
description: Render videos and pictures on a ComfyUI server that has the Continuity node pack (MiniMax H3, LTX 2.5, Krea 2, Ideogram 4, Qwen Image, Flux 2 Klein) with one command, cli/render.py. Use whenever the user asks to render, generate or make a clip, video, shot, still, picture or image "on ComfyUI", "on the lab", "with H3/Hailuo/MiniMax", "with LTX", "with Krea" and so on, or to test a render. Never hand-write ComfyUI workflow JSON or POST to /prompt for these models; this client builds the render correctly.
---

# Rendering with Continuity

The Continuity pack ships a client, `cli/render.py`, that renders on a running
ComfyUI over HTTP. The server builds the render from a family, a prompt and
optional pictures: it picks every model file, turns on the turbo settings and
checks the request before anything runs. You don't build workflows, pick
filenames or set sampler values.

## 1. Find the client and the server

- **Client:** `cli/render.py` inside the pack, at
  `<ComfyUI>/custom_nodes/<pack folder>/cli/render.py` (the folder is usually
  `ComfyUI-Continuity`). Find it with
  `find ~ -path '*custom_nodes/*/cli/render.py' 2>/dev/null | head`. The file
  is self-contained (Python 3.9+, standard library only), so a copy works from
  any machine. If none is on this machine, download that one file from
  https://github.com/roadmaus/ComfyUI-Continuity (`cli/render.py`).
- **Server:** `--url` or `$COMFY_URL`, default `http://127.0.0.1:8188`. If the
  user names a machine you don't have a URL for, ask. If it is reached through
  an SSH tunnel, open the tunnel first.

Always start with:

```
python3 cli/render.py families
```

This confirms that the server is reachable and has the pack. It lists which
families are ready (all their files installed) and what a fast render uses on
each. If it says the server has no `/continuity/render`, the pack is missing or
too old: tell the user rather than working around it.

## 2. Render

```
python3 cli/render.py h3 "a fluffy orange cat stretches on a sunny windowsill" --seconds 5
python3 cli/render.py krea2 "a tabby cat, studio portrait" --aspect 4:5
python3 cli/render.py h3 "@pic-1 turns and walks out of frame" --image cat.png
```

- **The first argument is the family id** from `families`. Video families
  (`h3`, `ltx25`) make clips, and image families make pictures. If the user
  says "MiniMax" or "Hailuo", that is `h3`.
- **Fast is the default, so keep it.** Add `--native` only when the user asks
  for full quality or a native render; native video can take an hour or more.
  `--quality draft|medium|good` changes the turbo step count.
- **Pictures:** `--image PATH[:AS]`, repeatable. A local file is uploaded. Cite
  it in the prompt as `@pic-1`, `@pic-2` (videos `@clip-N`, sounds `@snd-N`) in
  the order given. `AS` is `start`, `end`, `ref`, or a scope such as `style` or
  `person`; with nothing given, the first picture opens the shot. To reuse an
  earlier render, pass its name with ComfyUI's annotation:
  `--image "H3_00012_.png [output]"`.
- **Other flags:** `--seconds`, `--aspect 16:9|9:16|1:1|4:5|...`, `--edge`
  (short edge in px), `--seed`, `--model SLOT=FILE`, `--out DIR` (default
  `./renders`). See `--help`.
- **Run renders in the background with a long timeout.** A fast H3 clip takes
  minutes and a still takes under a minute. Progress lines go to stderr. On
  success stdout is the downloaded file path, one per line.

## 3. Writing the prompt

Write the shot as plain prose: subject, action, camera, light, sound. The pack
turns it into each model's own format, so don't write H3's section headers or
JSON captions yourself. For H3, describe the audio too (sounds, speech), since
it renders sound with the video. If a `minimax-h3-prompt` skill is available
and the user wants a carefully written H3 prompt, use it for the wording, then
pass its prose here.

## 4. When it refuses

The server answers with a sentence and the client exits non-zero. Act on the
sentence instead of trying another route:

- **No turbo LoRA, or several candidates:** pick one of the named files with
  `--turbo-lora NAME` (prefer the one whose step count matches the quality),
  or ask the user. Use `--native` only if the user accepts a slow render.
- **A family cannot render yet (missing files):** tell the user which slot is
  missing. Don't switch to another family without saying so.
- **A citation or picture problem** (for example `@pic-2` cited when only one
  picture was attached): fix the command.
- **A render that fails while running:** report the node and message it
  printed.

## Etiquette

- Queue one render at a time on a shared GPU. The client prints
  `queued, N ahead` when something is already running; tell the user rather
  than stacking more.
- Report the seed and the speed line (`turbo medium, 6 steps, <lora>`) with the
  result, so the render can be reproduced.
- Show or hand over the downloaded file path. Don't claim what the video looks
  like unless you have looked at it.
