# SimSat UI kit

A high-fidelity recreation of the SimSat operator surface — the dashboard
the founder stands behind in shots 3–8 of the demo video. It mirrors the
docs in `humanaiconvention/simsat` (`SUBMISSION_BRIEF.md`,
`SUBMISSION_CASEBOOK.md`, `VIDEO_SCRIPT.md`) — same column layout, same
encounter-window vocabulary, same six-gate viability filter — without
copying the production React/FastAPI implementation.

`index.html` is the canonical view: one operator session in progress, the
Rotterdam case loaded in the inspector pane (the architectural demo
case), with the action ribbon, encounter queue, scaffold/trust/operator
strip, and a viability-gate readout.

Components are split into small JSX files and wired up in the index. Use
the kit as a reference when designing any SimSat-shaped surface; treat
the index as the floor for what an operator screen looks like.
