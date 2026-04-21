## Architecture Explorer

The dashboard ships an **Architecture Explorer** (open via the header button) with side-by-side visualizations of every model the trust layer operates over. The intent is not to rank models but to make the comparison *legible* — so a reviewer can see at a glance whether the HAIC convention is being instantiated in a narrow or a broad slice of the current foundation-model landscape.

**HAIC variants — the convention's own trajectory:**

- `haic-v3-2b` — third-generation 2B-parameter HAIC baseline.
- `haic-v3-2b-q2`, `haic-v3-2b-q3`, `haic-v3-2b-q4` — quantization ablations of the v3 2B baseline, documenting the memory/accuracy tradeoff under the convention framing.
- `haic-v3-8b` — scaled-up 8B variant of the v3 line for reference.
- `haic-v4-2b`, `haic-v5-2b`, `haic-v6-2b`, `haic-v7-2b`, `haic-v8-2b` — iterative refinements of the 2B baseline as the convention evolved. Each revision encodes a specific change to how the model represents stimulus / interview / viability / receipt artifacts.

**Liquid AI — the activation-sparse trajectory:**

- `lfm2-8b` — Liquid Foundation Model, 8B parameters. Representative of the non-transformer / state-space design lineage Liquid AI has published on.
- `lfm2-8b-a1b` — Liquid's 8B / 1B-active activation-sparse variant. Interesting because activation sparsity is an obvious fit for on-orbit compute budgets where peak FLOPs are the constraint rather than parameter count.

**Open-weights frontier — the general landscape:**

- `gemma-3n-e2b`, `gemma3-4b`, `gemma4-2b`, `gemma4-e2b` — Google's Gemma 3 and Gemma 4 families at multiple scales, including the `e2b` efficient-2B variants.
- `llama3.1-8b` — Meta's Llama 3.1 8B, the current open-weights reference point.
- `ministral-3b` — Mistral's 3B dense model, included as a compact open-weights baseline.

### Why these three groups together

The comparison frames HAIC v3–v8 as **one evolutionary trajectory** (toward convention-compliant trust reasoning), LFM2 as a **parallel trajectory** (toward activation-sparse efficiency), and Gemma / Llama / Ministral as the **general open-weights landscape** the trust layer would have to cooperate with in any real deployment. All three trajectories are visualized with the same blocks-and-flows notation so structure, parameter budgets, and activation patterns can be compared at a glance.

For judges from Liquid AI specifically: LFM2-8b-a1b sitting next to `haic-v8-2b` in the Explorer is the shortest way to communicate that the HAIC convention is **not** a proprietary binding to any one architecture. It is a protocol for making decisions auditable on whichever architecture actually ships on-orbit — and LFM2-style activation sparsity is one of the more plausible candidates for that hardware envelope.
