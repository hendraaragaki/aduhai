// The client wires (OpenAI Chat Completions, Anthropic Messages) and the three upstream
// wires (OpenAI, Anthropic, Gemini): request translation, provider-answer parsing and
// client-wire rendering.
//
// The trial's providers answer every inference with the same text ("ANSWER ECHO{...}"), so
// the entry buffers one provider answer, parses it into {text, usage, stop} once, and renders
// it in the wire the client used. Buffering keeps the event order exact in both directions:
// an OpenAI usage chunk arrives after the finish chunk, while an Anthropic message_start
// needs the input tokens up front. Nothing here reads the clock or rolls dice: the same
// provider answer always renders the same bytes.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** One provider answer, parsed out of whichever wire carried it. */
export interface ProviderAnswer {
  text: string;
  usage: { input: number; output: number } | null;
  stop: string | null;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** The id stamped on rendered OpenAI answers. */
const OPENAI_ANSWER_ID = 'chatcmpl-aggregate';
/** The id stamped on rendered Anthropic answers. */
const ANTHROPIC_ANSWER_ID = 'msg_aggregate';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider, free: true })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT, free: true })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** Split a data URL into its media type and base64 payload; null when it is not one. */
function dataUrlOf(url: string): { mediaType: string; data: string } | null {
  const m = /^data:([^;,]+)?(;base64)?,(.*)$/s.exec(url);
  if (m === null) return null;
  return { mediaType: m[1] !== '' ? m[1] : 'image/png', data: m[3] };
}

/** One OpenAI content part as an Anthropic content block: text or an inline image; null for the rest. */
function blockOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = dataUrlOf(url);
    if (data !== null) return { type: 'image', source: { type: 'base64', media_type: data.mediaType, data: data.data } };
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/** The system prompt of an OpenAI body: every system message's text, joined with newlines. */
function openAISystem(body: Json): string {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const parts: string[] = [];
  for (const raw of messages) {
    if (!isRecord(raw) || raw.role !== 'system') continue;
    const text = textOf(raw.content);
    if (text !== '') parts.push(text);
  }
  return parts.join('\n');
}

/** The system prompt of an Anthropic body: its system string or blocks. */
function anthropicSystem(body: Json): string {
  return textOf(body.system);
}

/** An OpenAI stop field as a list: a string becomes one entry, an array stays one. */
function stopList(value: unknown): string[] {
  if (typeof value === 'string') return value === '' ? [] : [value];
  if (Array.isArray(value)) return value.filter((x): x is string => typeof x === 'string');
  return [];
}

/** Whether an OpenAI body asks for JSON mode. */
export function openAIJsonMode(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = anthropicSystem(body);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that names `model`: the system
 * messages become the system string, image_url parts become image blocks, tool_calls become tool_use
 * blocks, tool messages become tool_result blocks, tools and tool_choice are rewritten, and the output
 * cap, temperature, top_p, stop sequences and streaming pass through. JSON mode has no Anthropic
 * spelling: a caller that needs the provider to echo it must not route such a request here.
 */
export function openAIToAnthropic(body: Json, model: string): Json {
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') continue;
    if (role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: toolResultText(raw.content) }],
      });
      continue;
    }
    // Any other role reads as a user turn.
    const blocks: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') blocks.push({ type: 'text', text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        const block = blockOf(part);
        if (block !== null) blocks.push(block);
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }

  const out: Json = { model, messages };
  const system = openAISystem(body);
  if (system !== '') out.system = system;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopList(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** One Gemini content part for a text string. */
function geminiTextPart(text: string): Json {
  return { text };
}

/** One Gemini content part for an image: inline data for a data URL, a file reference otherwise. */
function geminiImagePart(url: string): Json {
  const data = dataUrlOf(url);
  if (data !== null) return { inlineData: { mimeType: data.mediaType, data: data.data } };
  return { fileData: { fileUri: url } };
}

/** One Gemini function declaration for a tool: name, description and its JSON schema. */
function geminiFunction(name: string, description: string, parameters: Json): Json {
  return { name, description, parameters };
}

/**
 * A client request (OpenAI or Anthropic wire) as a Gemini generateContent body. The model travels
 * in the URL, not the body. System goes to systemInstruction, user turns to user contents, assistant
 * turns to model contents, tool calls to functionCall parts, tool results to functionResponse parts,
 * tools to functionDeclarations, and the output cap, stop sequences and JSON mode to generationConfig.
 */
export function toGemini(api: 'openai' | 'anthropic', body: Json): Json {
  const system = api === 'openai' ? openAISystem(body) : anthropicSystem(body);
  const contents: Json[] = [];
  const declarations: Json[] = [];

  const pushUserParts = (parts: Json[]): void => {
    if (parts.length > 0) contents.push({ role: 'user', parts });
  };

  if (api === 'openai') {
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      const role = raw.role;
      if (role === 'system') continue;
      if (role === 'assistant') {
        const parts: Json[] = [];
        const text = textOf(raw.content);
        if (text !== '') parts.push(geminiTextPart(text));
        const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
        for (const call of calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          parts.push({
            functionCall: {
              name: typeof call.function.name === 'string' ? call.function.name : '',
              args: argumentsOf(call.function.arguments),
            },
          });
        }
        if (parts.length > 0) contents.push({ role: 'model', parts });
        continue;
      }
      if (role === 'tool') {
        contents.push({
          role: 'user',
          parts: [
            {
              functionResponse: {
                name: '',
                response: { result: toolResultText(raw.content) },
              },
            },
          ],
        });
        continue;
      }
      const parts: Json[] = [];
      if (typeof raw.content === 'string') {
        if (raw.content !== '') parts.push(geminiTextPart(raw.content));
      } else if (Array.isArray(raw.content)) {
        for (const part of raw.content) {
          if (!isRecord(part)) continue;
          if (part.type === 'text' && typeof part.text === 'string') parts.push(geminiTextPart(part.text));
          else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
            parts.push(geminiImagePart(part.image_url.url));
          }
        }
      }
      pushUserParts(parts);
    }
    if (Array.isArray(body.tools)) {
      for (const raw of body.tools) {
        if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
        declarations.push(
          geminiFunction(
            raw.function.name,
            typeof raw.function.description === 'string' ? raw.function.description : '',
            isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
          ),
        );
      }
    }
  } else {
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw) || !Array.isArray(raw.content)) continue;
      if (raw.role === 'assistant') {
        const parts: Json[] = [];
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push(geminiTextPart(block.text));
          else if (block.type === 'tool_use') {
            parts.push({
              functionCall: {
                name: typeof block.name === 'string' ? block.name : '',
                args: isRecord(block.input) ? block.input : {},
              },
            });
          }
        }
        if (parts.length > 0) contents.push({ role: 'model', parts });
        continue;
      }
      const parts: Json[] = [];
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push(geminiTextPart(block.text));
        else if (block.type === 'image' && isRecord(block.source)) {
          const src = block.source;
          if (src.type === 'base64' && typeof src.data === 'string') {
            parts.push({ inlineData: { mimeType: typeof src.media_type === 'string' ? src.media_type : 'image/png', data: src.data } });
          } else if (src.type === 'url' && typeof src.url === 'string') parts.push(geminiImagePart(src.url));
        } else if (block.type === 'tool_result') {
          parts.push({
            functionResponse: {
              name: '',
              response: { result: toolResultText(block.content) },
            },
          });
        }
      }
      pushUserParts(parts);
    }
    if (Array.isArray(body.tools)) {
      for (const raw of body.tools) {
        if (!isRecord(raw) || typeof raw.name !== 'string') continue;
        declarations.push(
          geminiFunction(
            raw.name,
            typeof raw.description === 'string' ? raw.description : '',
            isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          ),
        );
      }
    }
  }

  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config: Json = {};
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null;
  if (maxTokens !== null) config.maxOutputTokens = maxTokens;
  const stops = api === 'openai' ? stopList(body.stop) : stopList(body.stop_sequences);
  if (stops.length > 0) config.stopSequences = stops;
  if (api === 'openai' && openAIJsonMode(body)) config.responseMimeType = 'application/json';
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** Split streamed text into the pieces an SSE writer emits, 13 characters at a time like the providers. */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [''] : pieces;
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Parse an SSE body into its frames: each frame's event name (null when unnamed) and data lines joined. */
function sseFrames(text: string): Array<{ event: string | null; data: string }> {
  const frames: Array<{ event: string | null; data: string }> = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function jsonOrNull(text: string): Json | null {
  try {
    const value: unknown = JSON.parse(text);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** Parse an OpenAI provider answer (JSON body or SSE stream) into text, usage and stop reason. */
export function parseOpenAIAnswer(body: string, stream: boolean): ProviderAnswer {
  if (!stream) {
    const value = jsonOrNull(body);
    const choices = value !== null && Array.isArray(value.choices) ? value.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : {};
    const message = isRecord(choice.message) ? choice.message : {};
    const text = typeof message.content === 'string' ? message.content : textOf(message.content);
    const usage = isRecord(value?.usage)
      ? { input: numberOrZero(value.usage.prompt_tokens), output: numberOrZero(value.usage.completion_tokens) }
      : null;
    return { text, usage, stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : null };
  }
  let text = '';
  let usage: { input: number; output: number } | null = null;
  let stop: string | null = null;
  for (const frame of sseFrames(body)) {
    if (frame.data.trim() === '[DONE]') continue;
    const chunk = jsonOrNull(frame.data);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) usage = { input: numberOrZero(chunk.usage.prompt_tokens), output: numberOrZero(chunk.usage.completion_tokens) };
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') text += delta.content;
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
  }
  return { text, usage, stop };
}

/** Parse an Anthropic provider answer (JSON body or SSE stream) into text, usage and stop reason. */
export function parseAnthropicAnswer(body: string, stream: boolean): ProviderAnswer {
  if (!stream) {
    const value = jsonOrNull(body);
    const content = value !== null && Array.isArray(value.content) ? value.content : [];
    const usage = isRecord(value?.usage)
      ? { input: numberOrZero(value.usage.input_tokens), output: numberOrZero(value.usage.output_tokens) }
      : null;
    return { text: textOf(content), usage, stop: typeof value?.stop_reason === 'string' ? value.stop_reason : null };
  }
  let text = '';
  let input: number | null = null;
  let output: number | null = null;
  let stop: string | null = null;
  for (const frame of sseFrames(body)) {
    const data = jsonOrNull(frame.data);
    if (data === null) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      input = numberOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
      text += data.delta.text;
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) output = numberOrZero(data.usage.output_tokens);
    }
  }
  return { text, usage: input === null && output === null ? null : { input: input ?? 0, output: output ?? 0 }, stop };
}

/** Parse a Gemini provider answer (JSON body or SSE stream) into text, usage and stop reason. */
export function parseGeminiAnswer(body: string, stream: boolean): ProviderAnswer {
  if (!stream) {
    const value = jsonOrNull(body);
    const candidates = value !== null && Array.isArray(value.candidates) ? value.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
    const meta = isRecord(value?.usageMetadata) ? value.usageMetadata : null;
    const usage = meta !== null ? { input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount) } : null;
    return { text, usage, stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : null };
  }
  let text = '';
  let usage: { input: number; output: number } | null = null;
  let stop: string | null = null;
  for (const frame of sseFrames(body)) {
    const data = jsonOrNull(frame.data);
    if (data === null) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null) continue;
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const part of parts) if (isRecord(part) && typeof part.text === 'string') text += part.text;
    if (typeof candidate.finishReason === 'string') stop = candidate.finishReason;
    if (isRecord(data.usageMetadata)) {
      usage = { input: numberOrZero(data.usageMetadata.promptTokenCount), output: numberOrZero(data.usageMetadata.candidatesTokenCount) };
    }
  }
  return { text, usage, stop };
}

/** A provider stop reason as an OpenAI finish_reason. */
export function openAIStopOf(stop: string | null): string {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'length';
  if (stop === 'tool_calls' || stop === 'function_call' || stop === 'tool_use') return 'tool_calls';
  if (stop === 'content_filter' || stop === 'refusal' || stop === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** A provider stop reason as an Anthropic stop_reason. */
export function anthropicStopOf(stop: string | null): string {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'max_tokens';
  if (stop === 'tool_calls' || stop === 'function_call' || stop === 'tool_use') return 'tool_use';
  if (stop === 'content_filter' || stop === 'refusal' || stop === 'SAFETY') return 'refusal';
  return 'end_turn';
}

/** Render one parsed answer as an OpenAI Chat Completions body (JSON) or SSE stream. */
export function renderOpenAI(answer: ProviderAnswer, model: string, stream: boolean): string {
  const usage = answer.usage ?? { input: 0, output: 0 };
  if (!stream) {
    return JSON.stringify({
      id: OPENAI_ANSWER_ID,
      object: 'chat.completion',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: openAIStopOf(answer.stop), logprobs: null }],
      usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
    });
  }
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: OPENAI_ANSWER_ID,
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  let out = sseEvent(null, chunk({ role: 'assistant', content: '' }, null));
  for (const piece of splitText(answer.text)) out += sseEvent(null, chunk({ content: piece }, null));
  out += sseEvent(null, chunk({}, openAIStopOf(answer.stop)));
  out += sseEvent(null, {
    id: OPENAI_ANSWER_ID,
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  });
  out += 'data: [DONE]\n\n';
  return out;
}

/** Render one parsed answer as an Anthropic Messages body (JSON) or SSE stream. */
export function renderAnthropic(answer: ProviderAnswer, model: string, stream: boolean): string {
  const usage = answer.usage ?? { input: 0, output: 0 };
  if (!stream) {
    return JSON.stringify({
      id: ANTHROPIC_ANSWER_ID,
      type: 'message',
      role: 'assistant',
      model,
      content: [{ type: 'text', text: answer.text }],
      stop_reason: anthropicStopOf(answer.stop),
      stop_sequence: null,
      usage: { input_tokens: usage.input, output_tokens: usage.output },
    });
  }
  let out = sseEvent('message_start', {
    type: 'message_start',
    message: {
      id: ANTHROPIC_ANSWER_ID,
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: usage.input, output_tokens: 0 },
    },
  });
  out += sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
  for (const piece of splitText(answer.text)) {
    out += sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } });
  }
  out += sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
  out += sseEvent('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: anthropicStopOf(answer.stop), stop_sequence: null },
    usage: { output_tokens: usage.output },
  });
  out += sseEvent('message_stop', { type: 'message_stop' });
  return out;
}
