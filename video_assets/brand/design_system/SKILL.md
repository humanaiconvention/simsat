---
name: humanai-convention-design-system
description: HumanAI Convention design tokens, voice, and UI patterns. Calm, technical, dark-themed brand for the SimSat hackathon submission and humanaiconvention.com. Use whenever you build anything that should look like it ships under the HumanAI Convention name — slides, dashboards, landing pages, demo-video frames.
---

# HumanAI Convention — Design System

## What this is

A small, calm design system for the HumanAI Convention. The
hypothesis (the **Viability Convention**) is that AI systems should
be trained on a balance of synthetic generation and human lived
experience, governed by an explicit ethical framework.

This skill is the brand floor for everything HumanAI Convention
ships: the SimSat submission, humanaiconvention.com, and the
deductive framework write-ups. The full long-form context lives in
`README.md`. Read it before designing anything new.

## When to load this

- A user asks for "HumanAI Convention" anything
- A user asks for SimSat slides, dashboard, or website work
- The user pastes a brief that mentions Sentinel-2 / encounter window
  / six viability gates / on-orbit inference
- A user asks for a calm, dark, monochrome+yellow, receipts-driven
  technical brand and references this folder

## How to use it

1. **Tokens.** Link `colors_and_type.css` from any HTML file. It is
   the only token source — colors, type stack, spacing scale,
   radii, line weights, shadow. Use the CSS custom properties
   (`var(--bg-content)`, `var(--accent)`, `var(--font-sans)`, etc.)
   rather than hex literals.

2. **Voice.** Re-read "Content fundamentals" in `README.md` before
   writing copy. Numbers stand on their own. Honest negatives
   published next to wins. No exclamation marks, no emoji, no
   "harnessing AI." The five load-bearing brand sentences live in
   the README — reuse them verbatim when you need a sentence to
   sound right.

3. **Visual rules.** Re-read "Visual foundations" in `README.md`
   before laying out a frame. The yellow accent (`#fef3c7`) is
   never a button fill. Cards are bordered, not shadowed. Status
   pills are reserved for `accept` / `refine` / `defer` / `skip`.
   Sentinel-2 tiles render as specimens, with band-info in mono
   below.

4. **Components.** Use the recreations in `ui_kits/` as references:
    - `ui_kits/simsat/index.html` — operator console (encounter
      queue → inspector → receipts panel). Mirror this layout for
      any SimSat-shaped surface.
    - `ui_kits/website/index.html` — humanaiconvention.com landing
      (hero with Sentinel band, receipts grid, six-gate strip,
      footer). Mirror this hierarchy for any marketing surface.

5. **Slides.** Use `slides/index.html` for the SimSat demo-video
   title card (shot 2) and closing card (shot 9). The pattern is:
   small lockup top-left, center-stage title in Inter 200, Sentinel
   strip across the bottom with band-info in mono. Anywhere the
   brand needs a frame, that pattern is the floor.

6. **Iconography.** This brand barely uses icons. The mark does
   most of the work. If you need a small action icon, use
   Lucide-style 1.5-px stroke SVGs in `var(--fg-secondary)` and
   flag the addition — adopting an icon set is a brand decision the
   user must ratify.

## Token quick reference

```
--bg-cold-open  #000000   hero / video bookends
--bg-content    #0a0e27   dashboards, articles
--bg-elevated   #131838   cards on bg-content
--fg-primary    #dce6f5   body
--fg-secondary  #a0aec0   meta
--fg-bright     #ffffff   wordmark, rare emphasis
--accent        #fef3c7   warm yellow — single-number callouts only
--pos           #68d391   lift, gate-pass
--neg           #fc8181   regression, gate-block

--font-sans     Inter
--font-display  Inter (200/400/500)
--font-editorial Fraunces (website only, sparingly)
--font-mono     JetBrains Mono — every digit, every receipt

--r-sm 4   --r-md 8   --r-lg 12   --r-pill 999
--s-1 4 · --s-2 8 · --s-3 12 · --s-4 16 · --s-5 24 ·
--s-6 32 · --s-7 48 · --s-8 64
```

## What NOT to do

- Do not invent a logo. Use `assets/logo.svg`.
- Do not use the yellow as a button fill or a panel background.
- Do not add gradients behind data, decorative grain, hex grids, or
  glassy reflections.
- Do not add emoji. Not one.
- Do not write hype copy. The brand reports; it does not sell.
- Do not redraw the Sentinel-2 tiles or apply colour-grading. They
  ship true-colour, slightly cool, untouched.

## Files

| Path | What |
|---|---|
| `README.md` | Full long-form context — read first |
| `SKILL.md` | This file |
| `colors_and_type.css` | All tokens — link from every page |
| `assets/` | Logo SVGs, favicon, Sentinel tiles, architecture figure |
| `preview/` | Per-concept design-system cards |
| `ui_kits/simsat/` | SimSat operator console reference |
| `ui_kits/website/` | humanaiconvention.com reference |
| `slides/` | SimSat demo-video title + closing cards |
