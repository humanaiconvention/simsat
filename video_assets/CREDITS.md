# Demo video credits

Third-party assets used in `video_assets/simsat_demo_final.mp4`.

## Background music

**"moodtimeflow"** — composed by **ribhavagrawal**.

Source: [Pixabay — Sci-Fi Ambient · "moodtimeflow"](https://pixabay.com/music/ambient-sci-fi-moodtimeflow-194382/)

License: [Pixabay Content License](https://pixabay.com/service/license-summary/) —
royalty-free, free for commercial use, no attribution required by the
license. Credited here regardless because the artist's work added a
real production layer to this submission and we'd rather thank them
than not.

The track plays at base volume 0.45 throughout the 5:20 demo, with a
sidechain compressor (~8 dB ducking under voice, 400 ms release) so it
breathes up during the silent passages between voiceover beats.

## Imagery

- **Sentinel-2 imagery** (Port of Rotterdam, Houston Ship Channel, Nile
  Delta, Suez): contains modified Copernicus Sentinel data, freely
  available under the [Copernicus Open Access Hub Terms and Conditions](https://scihub.copernicus.eu/twiki/do/view/SciHubWebPortal/TermsConditions).
- **HumanAI Convention phi-with-dot mark and "Human AI / Convention"
  wordmark**: original to HumanAI Convention.

## Models

- **Liquid AI LFM2.5-VL-450M**: open-weight base model used for the
  encoder fine-tune. See [`LiquidAI/LFM2.5-VL-450M`](https://huggingface.co/LiquidAI/LFM2.5-VL-450M).
  SimSat's tuned LoRA adapter:
  [`HumanAIConvention/simsat-lfm25vl-450m-v3`](https://huggingface.co/HumanAIConvention/simsat-lfm25vl-450m-v3)
  (Apache-2.0).
- **Google Gemma-4-E2B**: General AI Track canonical backend. SimSat's
  tuned adapter: [`HumanAIConvention/simsat-gemma4-v11`](https://huggingface.co/HumanAIConvention/simsat-gemma4-v11)
  (Apache-2.0).

## Voiceover

Recorded by Ben Haslam, founder of HumanAI Convention.

## Design system

Generated via Claude Design from the HumanAI Convention brand
(humanaiconvention.com). The full design-system snapshot lives at
`video_assets/brand/design_system/`.

## Tooling

- **ffmpeg** (via [`imageio_ffmpeg`](https://pypi.org/project/imageio-ffmpeg/))
  — slide segmentation, sidechain ducking, video assembly.
- **Pillow** — slide rendering for shots 1, 3-8.
- **cairosvg** — SVG-to-PNG conversion of the brand mark.
- **Audacity** — voiceover capture.
