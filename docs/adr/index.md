# Architecture decisions

An architecture decision record (ADR) states one decision that shapes the code: the problem it
answers, what was decided, how it works in detail, what it costs, and what was rejected. The
roadmap says *what is planned*; an ADR says *why something built is the way it is*, so a later
change can argue with the reasons rather than rediscover them.

Numbered in the order they were accepted; never renumbered. A decision that is replaced keeps its
page and is marked superseded by the one that replaced it.

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-capability-gating.md) | Capability gating: a model declares what it exposes, a metric what it needs, and a metric that cannot run is a row with a reason | Accepted · 2026-09-25 |
| [0002](0002-verifying-a-foreign-model.md) | What can, and cannot, be verified about someone else's model: provenance, corpus ancestry, label space, a preprocessing fingerprint, and claims qualified on an unverified split | Accepted · 2026-09-25 |
