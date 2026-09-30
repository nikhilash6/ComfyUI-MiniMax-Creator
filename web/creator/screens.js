// Screens: content that goes onto a device screen after the render.
//
// The model is never shown the content. It is shown a tracker — a flat key
// colour with black markers — as one more reference picture, and after decode
// the screen pass finds the tracker in every frame and puts the real picture
// or clip on it, so the text on the screen is the file's own text. See
// `creator/screens/__init__.py` for the whole of it.
//
// What a screen may be, and the tracker's geometry, are `screenspec.js`, the
// pure mirror of the Python spec. This is the UI: the tool that adds one, the
// chips under the attachments, and the popover a chip opens.

import { el, icon, dismissable, placeNear } from "./dom.js";
import { t } from "./i18n.js";
import { uiSetting, viewUrl } from "./api.js";
import { stepperPill } from "./pills.js";
import { DEVICES, DEFAULT_DEVICE, DEFAULT_STYLE, FITS, WHEN_SHORT, CUTS, MAX_SCREENS,
         MAX_OFFSET_S, deviceOf, colourOf, cleanStyle, assignColours, trackerSize,
         screenProblem, trackerShapes } from "./screenspec.js";

// ---- the tracker, drawn --------------------------------------------------------------

/** This machine's tracker, off the settings cache. */
export const trackerStyle = () => cleanStyle(uiSetting("screen_tracker", DEFAULT_STYLE));

/**
 * The tracker as an SVG, `long` CSS pixels on its long side.
 *
 * Vector rather than a canvas: the same shapes Python rasterises, drawn at
 * whatever size the page asks for and sharp at any of them — a chip's 26
 * pixels and the settings page's specimen alike.
 */
export function trackerPicture({ pattern, colour }, size, long = 96) {
  const { width, height } = size;
  const scale = long / Math.max(width, height);
  const [r, g, b] = (colourOf(colour) ?? colourOf(DEFAULT_STYLE.colour)).fill;
  const shapes = trackerShapes(pattern, width, height).map(([layer, kind, points]) => {
    const fill = layer === "ink" ? "#000" : "#fff";
    if (kind === "rect") {
      const [x0, y0, x1, y1] = points;
      return `<rect x="${x0}" y="${y0}" width="${x1 - x0}" height="${y1 - y0}" fill="${fill}"/>`;
    }
    const [cx, cy, tri] = points;
    return `<polygon points="${cx},${cy - tri} ${cx + tri},${cy + tri} ${cx - tri},${cy + tri}" fill="${fill}"/>`;
  }).join("");
  const holder = document.createElement("span");
  holder.innerHTML = `<svg class="mmc-screen-tracker" aria-hidden="true" viewBox="0 0 ${width} ${height}" `
    + `width="${Math.round(width * scale)}" height="${Math.round(height * scale)}">`
    + `<rect width="${width}" height="${height}" fill="rgb(${r}, ${g}, ${b})"/>${shapes}</svg>`;
  return holder.firstElementChild;
}

// ---- the UI ---------------------------------------------------------------------------

const FIT_LABELS = { fill: "Fill", fit: "Fit", stretch: "Stretch" };
const FIT_TIPS = {
  fill: "Cover the whole screen, cropping what overhangs",
  fit: "Show all of it, with black bars where the shapes differ",
  stretch: "Stretch it to the screen's shape",
};
// How each fit looks in the preview: the CSS that does the same thing.
const FIT_CSS = { fill: "cover", fit: "contain", stretch: "fill" };
const SHORT_LABELS = { hold: "Hold last frame", loop: "Loop", first: "Back to first frame" };
const CUT_LABELS = { continue: "Keep playing", restart: "Start over" };

/** Which key colour screen `index` of a card is drawn in on this machine. */
export function screenStyle(screens, index) {
  const style = trackerStyle();
  const colours = assignColours(style.colour, Math.max(1, screens.length));
  return { pattern: style.pattern, colour: colours[Math.min(index, colours.length - 1)] };
}

const deviceName = (screen) => t(deviceOf(screen.device ?? DEFAULT_DEVICE)?.label ?? "Custom");
const fileStem = (name) => String(name ?? "").split("/").pop();

/** The glyph of a device's shape, for the device tiles. */
function deviceGlyph(size, long = 20) {
  const scale = long / Math.max(size.width, size.height);
  return el("span", { class: "mmc-screen-glyph" }, [el("span", { style: {
    width: `${Math.max(4, Math.round(size.width * scale))}px`,
    height: `${Math.max(4, Math.round(size.height * scale))}px`,
  } })]);
}

/**
 * The card's screens as a row of chips, or null when it has none.
 *
 * Each chip is the pair the whole feature is about: the tracker the model is
 * shown, and the file that replaces it. Clicking one opens its popover.
 *
 * @param {object} spec
 * @param {object} spec.state   the card (or still) whose `screens` these are
 * @param {() => void} spec.commit
 * @param {(kind:string) => Promise<{path:string, kind:string}|null>} spec.pick
 *   opens the picker for a replacement file
 */
export function screenChips({ state, commit, pick }) {
  const screens = state.screens ?? [];
  if (!screens.length) return null;
  return el("div", { class: "mmc-screens" }, screens.map((screen, index) => {
    const size = trackerSize(screen) ?? { width: 1920, height: 1080 };
    const problem = screenProblem(screen, index);
    const content = screen.content?.filename;
    return el("button", {
      class: `mmc-screen-chip${problem ? " mmc-screen-chip-bad" : ""}`,
      title: problem ?? t("{device}: the model is shown the tracker, and {file} goes on the screen after the render. Click to set it up.",
                          { device: deviceName(screen), file: fileStem(content) }),
      onclick: (event) => openScreenPopover(event.currentTarget, { state, index, commit, pick }),
    }, [
      trackerPicture(screenStyle(screens, index), size, 26),
      el("span", { class: "mmc-screen-arrow", "aria-hidden": "true", text: "→" }),
      content
        ? (screen.content.kind === "video"
            ? el("span", { class: "mmc-screen-thumb mmc-screen-thumb-clip" }, [icon("video", 14)])
            : el("img", { class: "mmc-screen-thumb", alt: "",
                          src: viewUrl(content, { preview: true }) }))
        : el("span", { class: "mmc-screen-thumb" }),
      el("span", { class: "mmc-screen-name" }, [
        el("span", { text: deviceName(screen) }),
        el("span", { class: "mmc-screen-file", text: content ? fileStem(content) : t("nothing picked") }),
      ]),
    ]);
  }));
}

/** Put a new screen on the card: pick what it shows, then open it to choose
 *  the device. Refuses past `MAX_SCREENS`, which the tool already says. */
export async function addScreen({ state, commit, pick, anchor }) {
  const screens = state.screens ?? (state.screens = []);
  if (screens.length >= MAX_SCREENS) return;
  const chosen = await pick(null);
  if (!chosen) return;
  screens.push({ device: DEFAULT_DEVICE,
                 content: { filename: chosen.path, kind: chosen.kind === "video" ? "video" : "image" } });
  commit();
  if (anchor?.isConnected) {
    openScreenPopover(anchor, { state, index: screens.length - 1, commit, pick });
  }
}

/** The tool that adds a screen, for a rail. */
export function screenTool({ state, commit, pick }) {
  const full = (state.screens?.length ?? 0) >= MAX_SCREENS;
  return el("button", {
    class: "mmc-tool",
    disabled: full ? true : undefined,
    title: full
      ? t("A shot can replace at most {max} screens.", { max: MAX_SCREENS })
      : t("Put a picture or a clip on a screen in the shot. The model draws a "
        + "marked placeholder there, and your file replaces it after the render, "
        + "with its text as sharp as the file's own."),
    onclick: (event) => addScreen({ state, commit, pick, anchor: event.currentTarget }),
  }, [el("span", { class: "mmc-tool-icon" }, [icon("res")]),
      el("span", { text: t("Add screen") })]);
}

/** A row of choices as one switch — the aspect popover's flip, reused. */
function flip(options, value, labels, onPick, tips = null) {
  return el("div", { class: "mmc-aspect-flip mmc-screen-flip", role: "radiogroup" },
    options.map((option) => el("button", {
      class: "mmc-flip-opt",
      "aria-pressed": option === value,
      title: tips ? t(tips[option]) : undefined,
      text: t(labels[option]),
      onclick: () => onPick(option),
    })));
}

/**
 * One screen's settings.
 *
 * Opens on the pair — what the model is shown beside what the viewer gets —
 * because that is the thing to understand about the feature, and because the
 * right half answers the fit question by showing it. Under it the device, then
 * the file and how it is fitted and timed.
 */
export function openScreenPopover(anchor, { state, index, commit, pick }) {
  const pop = el("div", { class: "mmc-pop mmc-screen-pop" });
  const body = el("div");

  const render = () => {
    const screens = state.screens ?? [];
    const screen = screens[index];
    if (!screen) { pop.remove(); return; }
    const size = trackerSize(screen) ?? { width: 1920, height: 1080 };
    const style = screenStyle(screens, index);
    const fit = screen.fit ?? FITS[0];
    const video = screen.content?.kind === "video";
    const problem = screenProblem(screen, index);
    const set = (patch) => { Object.assign(screen, patch); render(); commit(); };
    const drop = (key) => { delete screen[key]; render(); commit(); };

    const preview = (() => {
      const long = 112;
      const scale = long / Math.max(size.width, size.height);
      const box = { width: `${Math.round(size.width * scale)}px`,
                    height: `${Math.round(size.height * scale)}px` };
      const file = screen.content?.filename;
      const shown = file
        ? (video
            ? el("video", { class: "mmc-screen-content", muted: true, loop: true, autoplay: true,
                            playsinline: true, src: viewUrl(file), style: { objectFit: FIT_CSS[fit] } })
            : el("img", { class: "mmc-screen-content", alt: "", src: viewUrl(file, { preview: true }),
                          style: { objectFit: FIT_CSS[fit] } }))
        : null;
      return el("div", { class: "mmc-screen-pair" }, [
        el("figure", { class: "mmc-screen-side" }, [
          el("div", { class: "mmc-screen-frame", style: box }, [trackerPicture(style, size, long)]),
          el("figcaption", { text: t("The model is shown") }),
        ]),
        el("span", { class: "mmc-screen-arrow", "aria-hidden": "true", text: "→" }),
        el("figure", { class: "mmc-screen-side" }, [
          el("div", { class: "mmc-screen-frame mmc-screen-frame-content", style: box },
             shown ? [shown] : []),
          el("figcaption", { text: t("Replaced with") }),
        ]),
      ]);
    })();

    const devices = el("div", { class: "mmc-screen-devices", role: "radiogroup" },
      DEVICES.map((device) => {
        const tile = device.id === "custom"
          ? (trackerSize({ device: "custom", aspect: screen.aspect }) ?? device)
          : device;
        return el("button", {
          class: "mmc-screen-device",
          "aria-checked": (screen.device ?? DEFAULT_DEVICE) === device.id,
          onclick: () => {
            if (device.id === "custom" && !Array.isArray(screen.aspect)) {
              const current = trackerSize(screen) ?? size;
              screen.aspect = [current.width, current.height];
            }
            if (device.id !== "custom") delete screen.aspect;
            set({ device: device.id });
          },
        }, [deviceGlyph(tile), el("span", { text: t(device.label) })]);
      }));

    const rows = [
      el("div", { class: "mmc-screen-head" }, [
        el("span", { class: "mmc-pop-title", text: t("Screen {n}", { n: index + 1 }) }),
        el("button", {
          class: "mmc-screen-remove",
          title: t("Take this screen off the shot. The file stays where it is."),
          onclick: () => {
            screens.splice(index, 1);
            if (!screens.length) delete state.screens;
            commit();
            pop.remove();
          },
        }, [icon("trash", 15)]),
      ]),
      preview,
      devices,
    ];

    if ((screen.device ?? DEFAULT_DEVICE) === "custom") {
      const [w, h] = Array.isArray(screen.aspect) ? screen.aspect : [16, 9];
      const number = (value, apply) => el("input", {
        type: "number", class: "mmc-screen-ratio", min: "0.1", step: "0.1", value: String(value),
        onkeydown: (event) => event.stopPropagation(),
        onchange: (event) => {
          const next = Number(event.target.value);
          if (Number.isFinite(next) && next > 0) apply(next);
        },
      });
      rows.push(el("div", { class: "mmc-refine-row" }, [
        el("span", { class: "mmc-refine-label", text: t("shape") }),
        number(w, (next) => set({ aspect: [next, h] })),
        el("span", { class: "mmc-screen-by", text: ":" }),
        number(h, (next) => set({ aspect: [w, next] })),
      ]));
    }

    rows.push(el("div", { class: "mmc-refine-row" }, [
      el("span", { class: "mmc-refine-label", text: t("shows") }),
      el("span", { class: "mmc-screen-file", title: screen.content?.filename ?? "",
                   text: screen.content ? fileStem(screen.content.filename) : t("nothing picked") }),
      el("button", {
        class: "mmc-screen-change", text: t("Change…"),
        onclick: async () => {
          const chosen = await pick(screen.content?.kind ?? null);
          if (!chosen) return;
          set({ content: { filename: chosen.path, kind: chosen.kind === "video" ? "video" : "image" } });
        },
      }),
    ]));
    rows.push(el("div", { class: "mmc-refine-row" }, [
      el("span", { class: "mmc-refine-label", text: t("fit") }),
      flip(FITS, fit, FIT_LABELS, (next) => (next === FITS[0] ? drop("fit") : set({ fit: next })),
           FIT_TIPS),
    ]));
    if (video) {
      rows.push(el("div", { class: "mmc-refine-row" }, [
        el("span", { class: "mmc-refine-label", text: t("starts at") }),
        stepperPill({
          value: Number(screen.offset_s ?? 0), min: 0, max: MAX_OFFSET_S, step: 0.5, width: "52px",
          title: t("How far into the clip the screen starts playing."),
          format: (n) => `${n.toFixed(1)} s`,
          onChange: (next) => (next > 0 ? set({ offset_s: next }) : drop("offset_s")),
        }),
      ]));
      rows.push(el("div", { class: "mmc-refine-row mmc-screen-stack" }, [
        el("span", { class: "mmc-refine-label", text: t("if it runs out") }),
        flip(WHEN_SHORT, screen.when_short ?? WHEN_SHORT[0], SHORT_LABELS,
             (next) => (next === WHEN_SHORT[0] ? drop("when_short") : set({ when_short: next }))),
      ]));
      rows.push(el("div", { class: "mmc-refine-row" }, [
        el("span", { class: "mmc-refine-label", text: t("at a cut") }),
        flip(CUTS, screen.cuts ?? CUTS[0], CUT_LABELS,
             (next) => (next === CUTS[0] ? drop("cuts") : set({ cuts: next })),
             { continue: "The clip plays on through a cut the model makes inside the shot",
               restart: "The clip starts again from its start offset at each cut" }),
      ]));
    }
    const colour = colourOf(style.colour);
    rows.push(el("div", { class: "mmc-pop-note", text: [
      problem ?? "",
      t("Drawn as a {colour} {pattern} tracker — the settings page sets which.",
        { colour: t(colour.label).toLowerCase(),
          pattern: { grid: t("grid"), plus: t("crosses"), none: t("plain") }[style.pattern] }),
      index > 0 ? t("A second screen takes the next free key colour, so the two can be told apart.") : "",
    ].filter(Boolean).join(" ") }));
    body.replaceChildren(...rows);
  };

  render();
  pop.appendChild(body);
  document.body.appendChild(pop);
  placeNear(pop, anchor);
  dismissable(pop);
}
