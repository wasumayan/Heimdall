# Tech

Hound is a cognitive auditing framework designed to replicate how real expert auditors think, collaborate, and refine their understanding over time. Below we describe the main technical innovations that set Hound apart.

---

## 1. Dynamic Knowledge Graphs

![Dynamic graph example](./static/graph.png)

Hound builds dynamic, agent-chosen knowledge graphs that capture structure, flows, and invariants as “living” models of the codebase.

* **Flexible abstraction** – Nodes and edges are not hard-coded: the agent decides what to represent. Micro (functions, storage, modifiers), meso (authorization, pausing, staking subsystems), and macro (user flows, asset movements) all coexist in purpose-built graphs.
* **Agent-driven schema discovery** – In `analysis/graph_builder.py`, `GraphBuilder._discover_graphs` uses the “agent” LLM profile to propose graphs like `SystemArchitecture`, `StateMutationGraph`, `InterContractCallGraph`, and `AuthorizationRolesMap` based on the repository, files, and bundles.
* **Iterative refinement** – Graphs refine over multiple passes (`build(..., max_iterations=N)`), adding nodes, edges, and updates as more evidence is seen. Nodes track `observations` and `assumptions` with confidence and evidence links for later audit traceability.
* **No fixed representation** – Unlike static AST-/CFG-only approaches, graphs are constructed from code “cards” and bundling context, letting the agent emphasize what matters (e.g., staking lifecycles, fee paths, or cross-contract role gating).
* **Scales with size** – From a small library to a complex DeFi protocol, the granularity adapts: higher-level graphs first, then deeper graphs only where needed.

---

## 2. Iterative Hypothesis & Belief System

* **Targeted hypothesis formation** – Hound avoids the “spam hundreds of shallow guesses” approach. Instead, hypotheses are formed like a human auditor’s intuition: The most promising aspects are investigated first and the model thinks more deeply to form targeted hypotheses (see also 'dynamic model switching').
* **Belief refinement** – Hypotheses are tracked with confidence scores. As new code is explored, confidence is strengthened, weakened, or disproven. Evidence is attached, and findings evolve into either confirmed vulnerabilities or discarded ideas.
* **Long-horizon audits** – This design means audits scale with runtime: a day-long audit may surface basic issues, but a week-long audit accumulates richer understanding and deeper bugs, just like human auditors.

---

## 3. Dynamic Model Switching

* **Junior ↔ Senior auditor analogy** – A lightweight “junior” model (e.g. GPT-5-nano) can work through exploration steps and call on a “senior” model (e.g. GPT-5 or Claude Opus) for deeper guidance, strategic reasoning, or hypothesis formation.
* **Separation of duties** –

  * *Exploration model* → loads graphs, reads code, notes invariants and observations.
  * *Guidance model* → performs “deep think” passes that identify vulnerabilities and propose next directions.
* **Efficiency with quality** – This lets Hound keep costs low while still accessing heavyweight reasoning where it matters most.

---

## 4. Dynamic Multi-Agent Collaboration

* **Parallel or serial workflows** – Teams of agents can be assembled with heterogeneous models (e.g. GPT-5-nano + GPT-5, Claude Sonnet + Claude Opus) working on the same graph.
* **Shared knowledge base** – Graphs and hypotheses are stored in concurrent-safe stores, enabling agents to collaborate without overwriting each other.
* **Team orchestration** – Agents can work:

  * *In parallel* – covering different subsystems simultaneously.
  * *In series* – where one agent’s findings are reviewed or refined by another.
* **QA by design** – Final checks can be performed by a separate “QA agent” to ensure findings are consistent and well-supported.

---

## 5. Pipeline & Visualization

Hound’s graph pipeline is intentionally simple and observable:

* **Ingestion** (`commands/graph.py` → `RepositoryManifest`, `AdaptiveBundler`) – Repositories are turned into “cards” and adaptive bundles to give the agent contextual chunks without heavy preprocessing.
* **Discovery & Build** (`analysis/graph_builder.py`) – The “agent” profile proposes which graphs to build; the “graph” profile incrementally constructs them, iteration by iteration, storing `graph_*.json` plus `card_store.json` for provenance.
* **Interactive Viz** (`visualization/dynamic_graph_viz.py`) – Generates a dark-themed HTML with a graph selector, node-type filters, timeline, and node detail panel. The UI is designed for auditors to pivot across concerns quickly.
* **CLI UX** (`commands/graph.py`) – Rich progress, iteration status, and a summary table (graph name, nodes, edges, focus). Optional `--files` lets you scope to a whitelist.

Example usage:

```
hound graph build /path/to/repo --files "src/Foo.sol,src/Bar.sol" --graphs 2 --iterations 3 --visualize
```

## 6. Visualization & Reporting

* **Interactive graph exports** – Architecture, call, state-mutation, and flow graphs render to HTML with filtering, timelines, and source-card previews.
* **Professional reporting** – `analysis/report_generator.py` turns observations, confirmed issues, and context into a structured report.
* **Traceable findings** – Observations and conclusions link back to nodes and cards, preserving the line of evidence from code → graph → finding.
