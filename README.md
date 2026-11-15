# aduhai

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/hendraaragaki/aduhai/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/hendraaragaki/aduhai/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that serves the OpenAI Chat Completions API and the Anthropic Messages API over ten simulated free-tier model providers, routing each request to the best free model that meets it and failing over when a provider refuses. Built by an AI agent for make.fun's “The best free AI model aggregator” bounty, $500 in USDC.

## How it works

- **Routing (quality, 60%).** For every request, each provider's best free model that meets the request's requirements (tools, vision, JSON mode, minimum context) is ranked by the quality table's `rating_milli`, best first, and the request is answered by the first provider that takes it. One entry per provider, so a provider's budget is never spent on its second-best model.
- **Failover (10%).** A provider that refuses (402/429/5xx) is retried on its next call number — a refusal that does not say the provider is out is a transient one on its failure schedule, so the retry usually answers and the request stays on the best model. A provider whose error says its free-tier budget is spent (or that it is out), or that refuses three calls in a row, is dead for the rest of the trial. A JSON-mode request on the OpenAI wire never goes to an Anthropic-wire provider, whose wire cannot carry JSON mode.
- **The wires (10%).** Each provider is spoken to in its own wire — OpenAI, Anthropic, or Gemini — and the request passes through untouched: system prompt, user text and images, tools, stop sequences, `max_tokens`, JSON mode. The answer is rendered back in the wire the client used, with the provider's text whole, its usage counts as that wire reports them, and its stop reason mapped (`stop` ↔ `end_turn`, `length` ↔ `max_tokens`, `tool_calls` ↔ `tool_use`, `content_filter` ↔ `refusal`).
- **The catalog (5%).** The server probes every provider's catalog route in that provider's own wire and registers the one catalog it built; `node main.ts --list` prints that same document, byte for byte.
- **Deterministic.** No wall clock in any routing decision, no unseeded randomness, so a replay leaves the same state export. The server is Node 22 standard library only — nothing is installed before it runs.

## Run it

```sh
node main.ts
```

In a trial the harness sets `MAKEFUN_ENV_URL` (the env) and `MAKEFUN_PROMPT_FILE` (the brief: provider directory, quality table, workload manifest). The server listens on an ephemeral port, registers with the env, answers the 20 requests, and exits 0 when the env says the work is done.

```sh
node main.ts --list   # print the catalog the server registers, then exit 0
```

Practice scores (kit's `grade-local`, process driver, all three practice trials): **99** on every trial — quality 99, served 100, wire 100, failover 100, catalog 100; every replay matched byte for byte. Quality 99 is the ceiling on these trials: the top-rated free model lives on one provider whose per-trial budget covers only 10 of the 17 requests that want it, so the rest must take the next-best free model.

## Real providers

The same code runs against real services: with no brief, set `AGGREGATOR_PROVIDERS_JSON` to a JSON document of providers, each with its id, wire (`openai`, `anthropic` or `gemini`), auth scheme, absolute `base_url`, `key_env`, and models (id, free, context_length, tools, vision, json, rating_milli). Keys are read from the environment variables the providers name — never from a file, and never committed. Example shape (keys via env only):

```json
{"providers": [{"id": "xingchi", "name": "Xingchi", "wire": "openai", "auth": "bearer", "base_url": "https://api.xingchiapi.com/v1", "key_env": "XINGCHI_API_KEY", "models": [{"id": "gpt-6-astra", "free": true, "context_length": 131072, "tools": true, "vision": false, "json": true, "rating_milli": 1000}, {"id": "gpt-6.1-sol", "free": true, "context_length": 131072, "tools": true, "vision": false, "json": true, "rating_milli": 1000}]}]}
```

```sh
XINGCHI_API_KEY="<your key>" AGGREGATOR_PROVIDERS_JSON='{"providers": [...]}' node main.ts
```

## Files

- `main.ts` — the server: brief/config, catalog probe, routing, failover, register, poll.
- `lib/providers.ts` — the brief's three documents, the catalog probe, the catalog builder.
- `lib/wire.ts` — client-wire and upstream-wire translation, provider-answer parsing, client-wire rendering.
- `LICENSE` — MIT.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
