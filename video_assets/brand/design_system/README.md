# HumanAI Convention — Design System

A small, calm, technical design system for the HumanAI Convention — a
project to enable mass human flourishing through robust, ethical
training data. The hypothesis (the **Viability Convention**) is that
AI systems should be trained on a balance of synthetic generation and
human lived experience, governed by an explicit ethical framework.

This repo is the brand floor for everything HumanAI Convention ships:
the SimSat hackathon submission, humanaiconvention.com, and the
deductive framework write-ups.

---

## Sources used to assemble this system

The reader is not assumed to have access to any of these — they are
recorded so a future maintainer (or the user, with credentials) can
re-derive any decision.

- **Brand spec** — pasted by the project owner in the originating
  brief. Mark, palette, and tone notes flow from there verbatim.
- **Mark + favicon** — `uploads/logo.svg`, `uploads/favicon.svg`,
  `uploads/logo_video.png` (now copied into `assets/`).
- **SimSat repository** —
  [`humanaiconvention/simsat`](https://github.com/humanaiconvention/simsat)
  on `main`. Read in tree:
    - `README.md`, `SUBMISSION_BRIEF.md`, `SUBMISSION_TLDR.md`,
      `SUBMISSION_ABSTRACT.md`, `SUBMISSION_CASEBOOK.md`,
      `VIDEO_SCRIPT.md` for voice + headline numbers.
    - `submission_assets/*.png` — the four pinned operator-reviewed
      Sentinel-2 tiles. Four were pulled into `assets/` and used in
      the slides UI kit.
    - `fig/architecture_diagram.png` — the published architecture
      figure, now at `assets/architecture_diagram.png`.
- **Website repository** —
  [`humanaiconvention/humanaiconvention`](https://github.com/humanaiconvention/humanaiconvention)
  on `main` (private). Read for type stack confirmation
  (`web/src/App.css`): Inter for UI, Fraunces for editorial register,
  JetBrains Mono for receipts.
- **Hugging Face** — model weights live at
  [`HumanAIConvention`](https://huggingface.co/HumanAIConvention),
  Apache-2.0. Not bundled here.

If a future maintainer needs to refresh the system, those four
locations are the canonical sources in priority order: brand spec >
SimSat repo > website repo > the favicon SVG itself.

---

## What's in this folder

| File / folder | Purpose |
|---|---|
| `README.md` | this file — context, fundamentals, foundations, iconography, index |
| `SKILL.md` | machine-readable summary so this folder loads as an Agent SKill |
| `colors_and_type.css` | the only token file — colors, type, spacing, radii, shadows |
| `assets/` | logo SVGs, favicon, the four Sentinel-2 tiles, architecture figure |
| `fonts/` | empty — Inter / Fraunces / JetBrains Mono load from Google Fonts (see caveat) |
| `preview/` | one-card-per-concept HTML files registered as the Design System tab cards |
| `ui_kits/simsat/` | high-fidelity recreation of the SimSat dashboard surface |
| `ui_kits/website/` | high-fidelity recreation of humanaiconvention.com hero + sections |
| `slides/` | the two title/closing cards for the SimSat demo video, plus a stat-card pattern |

`uploads/` and `submission_assets/` (now empty) are residue from the
import step and can be left untouched or deleted by hand.

---

## Content fundamentals

The voice is **calm, technical, confident**. It does not sell. It
reports.

**What the brand sounds like (verified against `SUBMISSION_TLDR.md`,
`SUBMISSION_BRIEF.md`, and `VIDEO_SCRIPT.md`):**

- Numbers stand on their own. The phrase is `+68.8 pp`, not
  "a massive +68.8 pp lift." Receipts replace adjectives.
- Sentences end on the fact, not on a feeling. "All three negatives
  are published in the methodology doc." is a complete sentence.
- Disclaimers and negatives are published next to wins. The brand's
  phrase for this is "honest negatives published next to wins." Use
  it.
- "We" for the team, "you" for the operator/reader. First person
  singular only in voiceover ("I'm Ben Haslam"). Never "us"-vs-"them".
- Em-dashes set off a clarification — they do not signal excitement.
- Casing: sentence case for headlines and section titles. ALL CAPS
  is reserved for short eyebrow labels (`OPERATOR ACTION`, `RUNTIME`)
  and constants (`AGPL-3`, `JSON`).
- No exclamation marks. No "revolutionary," "groundbreaking,"
  "game-changing," "unleash," "supercharge," or any verb-as-headline
  hype.
- No emoji. Not one. The brief is explicit and SimSat's docs hold
  the line.
- Quotation marks are real (`"`). Smart quotes are fine in long-form
  prose, ASCII quotes are fine in code blocks.

**When in doubt, fewer words.** A title card that just says
*"+68.8 pp action lift"* over a Sentinel band is more on-brand than
the same card with a paragraph beneath it.

**Patterns to copy verbatim when you need a sentence to sound
right:**

- "On-orbit inference, continually refined by operator-labelled JSON
  within uplink bandwidth parameters."
- "Six viability gates govern any continual learning that runs on
  top."
- "What it can ferry is JSON."
- "The architecture is the contribution; the satellite is the test
  case."
- "On-orbit improvement is regime-shift re-adaptation, not monotonic
  improvement on fixed data."

These are the brand's load-bearing sentences. Reuse them.

**What to avoid in copy:**

- Pre-checked consent boxes, auto-accepted terms, or any phrasing
  that pretends a human approved something they didn't. The brand is
  about consent receipts; never write copy that violates that.
- Generic AI / space cliches: "the future of," "harnessing AI,"
  "reaching for the stars," "powering the next generation."
- Numbers without units. `0.844` is a measurement; `+68.8 pp` is a
  delta; `48.78%` is a fraction. Mixing them is sloppy.

---

## Visual foundations

**Colors.** Two backgrounds, one accent, two semantic. That's the
whole palette.

| Token | Hex | Use |
|---|---|---|
| `--bg-cold-open` | `#000000` | hero / cold-open frames, video bookends |
| `--bg-content`   | `#0a0e27` | content frames, dashboards, articles |
| `--bg-elevated`  | `#131838` | cards on top of `--bg-content` |
| `--fg-primary`   | `#dce6f5` | body text |
| `--fg-secondary` | `#a0aec0` | meta / captions |
| `--fg-bright`    | `#ffffff` | wordmark + the rare emphasis |
| `--accent`       | `#fef3c7` | the warm yellow — single-number callouts, headlines |
| `--pos`          | `#68d391` | lift, gate-pass, positive deltas |
| `--neg`          | `#fc8181` | regression, gate-block |

The yellow is **never** a button fill. It paints text and the
occasional thin underline. Backgrounds stay dark.

**Type.** Inter for everything UI; Fraunces for editorial pull-quotes
on the website only; JetBrains Mono for any digit, JSON blob, or
receipt. The display lockup uses three weights of Inter (200 / 400 /
300) — see `wordmark` in `colors_and_type.css`.

**Spacing.** 4-px grid. The most-used rhythm is `s-4 / s-5 / s-7`
(16 / 24 / 48 px) — body, between-paragraph, between-section.

**Backgrounds & imagery.** Full-bleed dark fields. The hero/title
frames are pure black with a single Sentinel-2 band as the only
texture (clipped strip, no overlay). Content frames sit on
`#0a0e27`. There are **no decorative gradients** — the website's
homepage uses a soft radial blue/teal aurora behind the lockup, and
that's the only place a gradient appears in the system; it never
shows up behind data. No grain, no noise, no subtle textures, no hex
grids.

**Animation.** The reference is shot 1 of the SimSat demo video:
phi-with-dot fades in over 2 s, holds, then the wordmark crossfades
in below. Easing is `cubic-bezier(.22,.9,.24,1)` — slow start, soft
land. Durations cluster at 600 ms for entrance, 160 ms for hover.
No bounces. No spring physics. No looping ambient motion.

**Hover states.** A border or underline brightens by ~12-16% lumin-
ance; fills do not change. `border-color: rgba(255,255,255,0.18) →
rgba(255,255,255,0.32)` is the canonical step.

**Press states.** A 1-px translate-y down and a 4 % alpha drop on
the same border. Never shrink-on-press: this brand isn't tactile,
it's analytical.

**Borders & lines.** All dividers are `rgba(220,230,245,0.08-0.16)`.
Solid white lines never appear. Cards are bordered, not shadowed.

**Cards.** `--bg-elevated` (`#131838`), 1-px `--line-soft` border,
`var(--r-md)` radius (8 px). Inner highlight via
`box-shadow: 0 1px 0 rgba(255,255,255,0.04) inset` only. No drop
shadows on cards over content; drop shadows are reserved for the
very rare popover (`--shadow-pop`).

**Corner radii.** 4 px (chips, code), 8 px (cards, inputs), 12 px
(modals). Pills (`999px`) only on status badges that contain a single
short label.

**Transparency & blur.** A 6-px backdrop blur is permitted on the
nav bar of the website only. Anywhere else, opacity is achieved
through colour mixing, not `filter: blur`.

**Layout rules.** Center column on title cards, max-width ~720 px
for body prose, 12-col fluid grid on dashboards. The mark is always
optically centered above the wordmark with a gap of `0.42 ×
mark-height`.

**Imagery vibe.** Sentinel-2 RGB tiles render true-colour and
slightly cool. No grading is applied. They are presented as
specimens, not as backdrops — usually framed inside a thin border
strip with the bandinfo (sensor, cloud %, datetime) below in mono.

---

## Iconography

This brand barely uses icons.

- The **mark** (`assets/logo.svg`) is the most prominent symbol and
  it does most of the work. White on dark, never re-coloured. Never
  redrawn freehand — the SVG is geometric and analytic-locked
  (golden-ratio inner span). When you need it small, use
  `assets/favicon.svg`.
- **Status indicators** in the SimSat surface are simple shapes —
  filled dot for `accept`, hollow circle for `defer`, a short bar for
  `refine`, an `×` for `skip`. They are drawn in CSS / inline SVG in
  `ui_kits/simsat/StatusGlyph.jsx`.
- **No icon font is bundled.** The website source confirms that
  Fraunces, Inter, and JetBrains Mono are the only web fonts loaded
  and there is no Lucide / Phosphor / Heroicons import in the
  pipeline.
- **Emoji are not used.** Anywhere.
- **Unicode-as-icon** appears once: the `phi` letter `φ` is used as
  a fallback when SVG is unavailable (e.g. in plain-text email
  signatures). Otherwise text and the mark carry the load.

If a screen genuinely needs a small action icon (e.g. a `download`
or `external-link`), use Lucide-style 1.5-px stroke SVGs in
`var(--fg-secondary)` and **flag the addition** — the brand has no
established icon set and adopting one is a brand decision the user
should ratify.

---

## Caveats — please review

- **Font files** are not bundled. Inter, Fraunces, and JetBrains Mono
  load from Google Fonts at runtime, mirroring the website. If this
  system needs to ship offline (a printable PDF, a sandboxed video
  render), please drop the static `.woff2` files into `fonts/` and
  swap the Google import for a `@font-face` block.
- **Phi-with-dot mark** in the slides and UI kit is loaded from the
  source SVG; it is never redrawn.
- **Iconography is intentionally bare.** If the project needs a
  fuller icon set, that's a brand decision — flag it for me and I'll
  introduce one.

---

## Index — where to look for what

- I want the tokens → `colors_and_type.css`
- I want the brand voice → "Content fundamentals" above
- I want the rules → "Visual foundations" above
- I want the cards in the Design System tab → `preview/*.html`
- I want a SimSat dashboard mock → `ui_kits/simsat/index.html`
- I want a humanaiconvention.com mock → `ui_kits/website/index.html`
- I want the demo-video title and closing cards → `slides/index.html`
- I want this as a portable Agent Skill → `SKILL.md`
