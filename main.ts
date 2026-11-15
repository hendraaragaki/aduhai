// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// Serves the OpenAI Chat Completions API and the Anthropic Messages API (JSON and SSE) over
// the providers the trial's brief names, and the same code serves real providers when the
// brief is absent (see README: AGGREGATOR_PROVIDERS_JSON).
//
// Routing: for each request, every provider's best free model that meets the request's
// requirements is ranked by the quality table's rating_milli, best first, and the request is
// answered by the first provider that takes it. A provider that refuses (402/429/5xx) is
// failed over to the next candidate for that request; a provider whose error says its
// free-tier budget is spent (or that it is out), or that refuses three calls in a row, is
// dead for the rest of the trial. A JSON-mode request on the OpenAI wire never goes to an
// Anthropic-wire provider, whose wire cannot carry JSON mode.
//
// The wires: each provider is spoken to in its own wire (OpenAI, Anthropic, Gemini), the
// request passed through untouched — system prompt, user text and images, tools, stop
// sequences, max_tokens, JSON mode — and its answer is rendered back in the wire the client
// used, with the provider's text whole, its usage counts as that wire reports them, and its
// stop reason mapped. Nothing here reads the wall clock for a routing decision and nothing
// rolls dice, so a replay leaves the same state export.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  resolveKey,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type ProviderDirectory,
  type RatingsFile,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonMode,
  openAIModelList,
  openAIToAnthropic,
  parseAnthropicAnswer,
  parseGeminiAnswer,
  parseOpenAIAnswer,
  renderAnthropic,
  renderOpenAI,
  toGemini,
  type ProviderAnswer,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** The most milliseconds one client request may spend on upstream calls (a safety bound; the trial's sim answers in milliseconds, so this never binds there). */
const REQUEST_DEADLINE_MS = 25000;
/** How many refusals in a row mark a provider dead when its error does not say why. */
const CONSECUTIVE_FAILURES_TO_DIE = 3;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
/** Real-provider config (no brief): a JSON document of providers, see README. */
const REAL_PROVIDERS_JSON = process.env.AGGREGATOR_PROVIDERS_JSON ?? '';

/** The brief, or the real-provider config, or an empty directory when neither is given. */
function loadSources(): { directory: ProviderDirectory; ratings: RatingsFile | null; workload: { manifest: Array<{ id: string; requires: Requirement }> } | null } {
  const brief = readBrief(PROMPT_FILE);
  if (brief.error === null && brief.directory.providers.length > 0) {
    return { directory: brief.directory, ratings: brief.ratings, workload: brief.workload };
  }
  if (REAL_PROVIDERS_JSON !== '') {
    try {
      const doc: unknown = JSON.parse(REAL_PROVIDERS_JSON);
      if (isRecord(doc) && Array.isArray(doc.providers)) {
        const providers: ProviderDirectory['providers'] = [];
        const ratings: RatingsFile = { schema: 'free-model-aggregator-01/ratings/v1', ratings: [], best: [] };
        for (const raw of doc.providers) {
          if (!isRecord(raw) || typeof raw.id !== 'string') continue;
          const wire = raw.wire === 'anthropic' || raw.wire === 'gemini' ? raw.wire : 'openai';
          const auth = raw.auth === 'x-api-key' || raw.auth === 'x-goog-api-key' || raw.auth === 'none' ? raw.auth : 'bearer';
          const base = typeof raw.base_url === 'string' && raw.base_url !== '' ? raw.base_url : typeof raw.path === 'string' ? raw.path : '';
          providers.push({
            id: raw.id,
            name: typeof raw.name === 'string' ? raw.name : raw.id,
            wire,
            auth,
            base,
            key_env: typeof raw.key_env === 'string' ? raw.key_env : '',
            key: typeof raw.key === 'string' ? raw.key : '',
          });
          if (Array.isArray(raw.models)) {
            for (const m of raw.models) {
              if (!isRecord(m) || typeof m.id !== 'string') continue;
              ratings.ratings.push({
                model: m.id,
                provider: raw.id,
                rating_milli: typeof m.rating_milli === 'number' ? m.rating_milli : 1000,
                context_length: typeof m.context_length === 'number' ? m.context_length : null,
                tools: m.tools === true,
                vision: m.vision === true,
                json: m.json === true,
                free: m.free !== false,
              });
            }
          }
        }
        return { directory: { schema: 'free-model-aggregator-01/providers/v1', providers }, ratings, workload: null };
      }
    } catch {
      // A malformed real-provider config falls through to the empty directory.
    }
  }
  return { directory: brief.directory, ratings: brief.ratings, workload: brief.workload };
}

const sources = loadSources();
const directory = sources.directory;
const ratings = sources.ratings;
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Providers a refusal has shown to be out of budget, or that refused three calls in a row. */
const deadProviders = new Set<string>();
/** Refusals in a row per provider, reset by a success. */
const consecutiveFailures = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  probes = await Promise.all(directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * The providers that can serve one request, each with its best free model that meets the
 * requirements, ranked best first (rating, then provider id, then model id, so the order is
 * a pure function of the brief and the probes). A JSON-mode request on the OpenAI wire skips
 * Anthropic-wire providers: their wire cannot carry JSON mode, so the provider would echo it
 * absent.
 */
function candidatesFor(requires: Requirement, needsJsonWire: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (needsJsonWire && probe.provider.wire === 'anthropic') continue;
    let best: Candidate | null = null;
    for (const model of probe.models) {
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      const rating = row?.rating_milli ?? 0;
      if (best === null || rating > best.rating || (rating === best.rating && model.id < best.model.id)) {
        best = { probe, model, rating };
      }
    }
    if (best !== null) out.push(best);
  }
  out.sort((a, b) => b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : 1) || (a.model.id < b.model.id ? -1 : 1));
  return out;
}

/** Whether a provider's error message says its free tier is gone (budget spent, or out). */
function isExhausted(message: string): boolean {
  return /budget is spent/i.test(message) || /\bis out\s*$/i.test(message.trim()) || /quota exceeded/i.test(message);
}

function noteSuccess(providerId: string): void {
  consecutiveFailures.set(providerId, 0);
}

function noteFailure(providerId: string, message: string): void {
  const n = (consecutiveFailures.get(providerId) ?? 0) + 1;
  consecutiveFailures.set(providerId, n);
  if (isExhausted(message) || n >= CONSECUTIVE_FAILURES_TO_DIE) {
    deadProviders.add(providerId);
    log(`provider ${providerId} is dead for this trial (${message.slice(0, 100) || `refused ${n} calls in a row`})`);
  }
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = sources.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this fallback is a
  // guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The message and code out of a provider's error body, in any of the three wires' envelopes. */
function errorOf(text: string): { code: string; message: string } {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error)) {
      const error = value.error;
      const message = typeof error.message === 'string' ? error.message : '';
      const code = typeof error.code === 'string' ? error.code : typeof error.type === 'string' ? error.type : typeof error.status === 'string' ? error.status : '';
      return { code, message };
    }
  } catch {
    // Not JSON: the raw text is the message.
  }
  return { code: '', message: text.slice(0, 200) };
}

interface UpstreamResult {
  ok: boolean;
  status: number;
  answer: ProviderAnswer | null;
  code: string;
  message: string;
}

/** One inference call to one provider, in that provider's own wire. */
async function callUpstream(candidate: Candidate, api: 'openai' | 'anthropic', body: Json, stream: boolean, deadlineMs: number): Promise<UpstreamResult> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, resolveKey(provider)),
  };
  if (stream) headers.accept = 'text/event-stream';
  let url: string;
  let payload: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    if (api === 'openai') {
      payload = { ...body, model: candidate.model.id };
      if (stream) payload.stream_options = { include_usage: true };
    } else {
      payload = anthropicToOpenAI(body, candidate.model.id);
    }
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    payload = api === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropic(body, candidate.model.id);
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${stream ? 'streamGenerateContent' : 'generateContent'}`;
    payload = toGemini(api, body);
  }
  const timeoutMs = Math.max(1000, deadlineMs - Date.now());
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(payload), signal: AbortSignal.timeout(timeoutMs) });
  } catch (err) {
    return { ok: false, status: 0, answer: null, code: 'fetch_error', message: (err as Error).message };
  }
  const text = await res.text();
  if (!res.ok) {
    const error = errorOf(text);
    return { ok: false, status: res.status, answer: null, code: error.code, message: error.message };
  }
  const answer =
    provider.wire === 'openai'
      ? parseOpenAIAnswer(text, stream)
      : provider.wire === 'anthropic'
        ? parseAnthropicAnswer(text, stream)
        : parseGeminiAnswer(text, stream);
  return { ok: true, status: res.status, answer, code: '', message: '' };
}

interface Served {
  status: number;
  contentType: string;
  text: string;
}

/**
 * Serve one client request: rank the candidates, call them in order until one answers, and
 * render that answer in the client's wire. A refusal fails over to the next candidate; a
 * provider that is out is never called again. A refusal that does not say the provider is
 * out is a transient one on its failure schedule, so the same provider is retried (its next
 * call number answers) before falling back: that keeps the request on the best model instead
 * of dropping it to the next tier. A 400/401 is a config error, never retried.
 */
async function serve(api: 'openai' | 'anthropic', body: Json): Promise<Served> {
  const stream = body.stream === true;
  const requires = requirementsFor(body);
  const needsJsonWire = api === 'openai' && openAIJsonMode(body);
  const candidates = candidatesFor(requires, needsJsonWire);
  const deadlineMs = Date.now() + REQUEST_DEADLINE_MS;
  for (const candidate of candidates) {
    if (deadProviders.has(candidate.probe.provider.id)) continue;
    for (let attempt = 1; attempt <= 3; attempt++) {
      if (Date.now() > deadlineMs) break;
      const result = await callUpstream(candidate, api, body, stream, deadlineMs);
      if (result.ok && result.answer !== null) {
        noteSuccess(candidate.probe.provider.id);
        const text = api === 'openai' ? renderOpenAI(result.answer, candidate.model.id, stream) : renderAnthropic(result.answer, candidate.model.id, stream);
        return { status: 200, contentType: stream ? 'text/event-stream; charset=utf-8' : 'application/json', text };
      }
      noteFailure(candidate.probe.provider.id, result.message);
      log(`failover: ${candidate.probe.provider.id}/${candidate.model.id} attempt ${attempt} answered HTTP ${result.status} ${result.code} ${result.message.slice(0, 120)}`);
      if (deadProviders.has(candidate.probe.provider.id)) break;
      if (result.status === 400 || result.status === 401) break;
    }
    if (Date.now() > deadlineMs) break;
  }
  const detail = candidates.length === 0 ? 'no free model meets this request' : 'every free model that meets this request refused it';
  return {
    status: 503,
    contentType: 'application/json',
    text: JSON.stringify(api === 'anthropic' ? anthropicError('overloaded_error', detail) : openAIError(detail, 'service_unavailable', null)),
  };
}

/** POST /v1/chat/completions or POST /v1/messages: one client request, answered in its own wire. */
async function infer(req: IncomingMessage, res: ServerResponse, api: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, api, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve(api, body);
  const headers: Record<string, string> = { 'content-type': served.contentType, 'content-length': String(Buffer.byteLength(served.text)) };
  if (served.contentType.startsWith('text/event-stream')) {
    headers['cache-control'] = 'no-cache';
    headers.connection = 'keep-alive';
  }
  res.writeHead(served.status, headers);
  res.end(served.text);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await infer(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await infer(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
