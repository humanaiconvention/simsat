# Encounter Service

The encounter planner now uses a deterministic WCLI-style decision stack:
- analytic scaffold score
- blended support score
- trust head
- trust-gated residual
- explicit `refine` action when trust is low but mission value is still high

Endpoints:
- `GET /encounter/policy`
- `GET /encounter/targets`
- `POST /encounter/targets`
- `PUT /encounter/targets/{target_id}`
- `DELETE /encounter/targets/{target_id}`
- `GET /encounter/windows`
- `POST /encounter/plan`
- `POST /encounter/evaluate`
- `GET /encounter/evaluations`
- `GET /encounter/decisions`
- `POST /encounter/decision/{decision_id}/materialize`

Behavior:
- enumerate future imaging windows from TLE propagation
- persist targets, decisions, sessions, and stimuli with atomic file writes
- rank windows with a scaffolded WCLI-style policy
- compare scaffold vs WCLI-trust planners on the same windows
- log accept/defer/refine distributions and materialization yield
- persist decision records and compact artifacts
- optionally materialize a chosen decision into the existing HAIC stimulus flow

Still out of scope:
- trained learned weights
- autonomous actuation
- benchmark/evaluation dashboards
