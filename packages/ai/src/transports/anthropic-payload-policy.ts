import type { BetaContextManagementConfig } from "@anthropic-ai/sdk/resources/beta/messages/messages.js";
import type { TextBlockParam } from "@anthropic-ai/sdk/resources/messages.js";
import type { Model } from "@openclaw/llm-core";
import { isRecord } from "@openclaw/normalization-core/record-coerce";
import { normalizeOptionalLowercaseString } from "@openclaw/normalization-core/string-coerce";
import { getAiTransportHost } from "../host.js";
import type { AnthropicContextManagementOptions } from "../provider-options.js";
import { isAnthropicOAuthApiKey } from "../providers/anthropic-auth-headers.js";
import {
  ANTHROPIC_CLAUDE_CODE_VERSION,
  resolveClaudeOpus5ModelIdentity,
  resolveClaudeSonnet55ModelIdentity,
  usesClaudeFable5MessagesContract,
} from "../providers/anthropic-model-contract.js";
import {
  ANTHROPIC_SERVER_SIDE_FALLBACK_BETA,
  ANTHROPIC_SERVER_SIDE_FALLBACKS,
} from "../providers/anthropic-server-fallback.js";
import { resolveCacheRetention } from "../providers/cache-retention.js";
import { sanitizeSurrogates } from "../utils/sanitize-unicode.js";
import {
  splitSystemPromptCacheBoundary,
  stripSystemPromptCacheBoundary,
  stripSystemPromptRelocatableBoundary,
} from "../utils/system-prompt-cache-boundary.js";
import { resolveProviderEndpoint, resolveProviderRequestCapabilities } from "./host-policy.js";
import { parsePositiveInteger } from "./transport-utils.js";

/** @deprecated Anthropic-family provider payload helper; do not use from third-party plugins. */
type AnthropicServiceTier = "auto" | "standard_only";

/** @deprecated Anthropic-family provider payload helper; do not use from third-party plugins. */
type AnthropicEphemeralCacheControl = {
  type: "ephemeral";
  ttl?: "1h" | "5m";
};

type AnthropicPayloadPolicyInput = {
  api?: string;
  baseUrl?: string;
  cacheRetention?: "short" | "long" | "none";
  cacheTtlPruning?: AnthropicContextManagementOptions["cacheTtlPruning"];
  contextWindow?: unknown;
  enableCacheControl?: boolean;
  enableServerCompaction?: boolean;
  extraParams?: Record<string, unknown>;
  provider?: string;
  serviceTier?: AnthropicServiceTier;
};

const ANTHROPIC_CACHE_CONTROL_LIMIT = 4;
const ANTHROPIC_COMPACT_THRESHOLD_MIN = 50_000;

/** @deprecated Anthropic-family provider payload helper; do not use from third-party plugins. */
type AnthropicPayloadPolicy = ReturnType<typeof resolveAnthropicPayloadPolicy>;

/** Resolve the Anthropic input-token trigger, including the API's minimum. */
function resolveAnthropicCompactThreshold(contextWindow: unknown, configured: unknown): number {
  const configuredThreshold = parsePositiveInteger(configured);
  if (configuredThreshold !== undefined) {
    return Math.max(ANTHROPIC_COMPACT_THRESHOLD_MIN, configuredThreshold);
  }
  const resolvedContextWindow = parsePositiveInteger(contextWindow);
  return Math.max(ANTHROPIC_COMPACT_THRESHOLD_MIN, Math.floor((resolvedContextWindow ?? 0) * 0.7));
}

/** Resolve the server-compaction gate and effective threshold for an Anthropic route. */
export function resolveAnthropicServerCompactionPlan(
  model: {
    provider?: unknown;
    api?: unknown;
    baseUrl?: string;
    contextWindow?: unknown;
  },
  extraParams?: Record<string, unknown>,
  apiKey?: string,
): { enabled: boolean; threshold?: number } {
  const enabled =
    extraParams?.anthropicServerCompaction === true &&
    !isAnthropicOAuthApiKey(apiKey) &&
    normalizeOptionalLowercaseString(model.api) === "anthropic-messages" &&
    isDirectAnthropicModel(model);
  return {
    enabled,
    ...(enabled
      ? {
          threshold: resolveAnthropicCompactThreshold(
            model.contextWindow,
            extraParams?.anthropicCompactThreshold,
          ),
        }
      : {}),
  };
}

export function isDirectAnthropicModel(
  model: { provider?: unknown; baseUrl?: string },
  env: { ANTHROPIC_BASE_URL?: string } = process.env,
): boolean {
  const baseUrl = model.baseUrl?.trim() || env.ANTHROPIC_BASE_URL?.trim();
  const endpointModel = baseUrl === model.baseUrl ? model : { ...model, baseUrl };
  const endpointClass = resolveProviderEndpoint(endpointModel).endpointClass;
  return (
    normalizeOptionalLowercaseString(model.provider) === "anthropic" &&
    (endpointClass === "anthropic-public" ||
      (endpointClass === "default" &&
        (!baseUrl || URL.parse(baseUrl)?.hostname === "api.anthropic.com")))
  );
}

// Proxies reject this first-party beta; callers additionally exclude OAuth requests.
export function supportsAnthropicServerSideFallback(model: Model<"anthropic-messages">): boolean {
  return (
    (usesClaudeFable5MessagesContract(model) ||
      resolveClaudeOpus5ModelIdentity(model) !== undefined ||
      resolveClaudeSonnet55ModelIdentity(model) !== undefined) &&
    isDirectAnthropicModel(model)
  );
}

export function isAnthropicServerToolClearingEnabled(
  model: { provider?: unknown; api?: unknown; baseUrl?: string },
  apiKey?: string,
): boolean {
  const resolvedApiKey = apiKey && getAiTransportHost().resolveSecretSentinel(apiKey);
  // Only a prepared direct API-key request can take ownership away from client pruning.
  return (
    Boolean(resolvedApiKey?.trim()) &&
    !isAnthropicOAuthApiKey(resolvedApiKey) &&
    normalizeOptionalLowercaseString(model.api) === "anthropic-messages" &&
    isDirectAnthropicModel(model)
  );
}

function isLongTtlEligibleEndpoint(baseUrl: string | undefined): boolean {
  if (typeof baseUrl !== "string") {
    return false;
  }
  const hostname = URL.parse(baseUrl)?.hostname;
  if (!hostname) {
    return false;
  }
  return (
    hostname === "api.anthropic.com" ||
    hostname === "aiplatform.googleapis.com" ||
    hostname === "aiplatform.us.rep.googleapis.com" ||
    hostname === "aiplatform.eu.rep.googleapis.com" ||
    hostname.endsWith("-aiplatform.googleapis.com")
  );
}

/** Resolve Anthropic cache-control marker retention for a request endpoint. */
export function resolveAnthropicEphemeralCacheControl(
  baseUrl: string | undefined,
  cacheRetention: AnthropicPayloadPolicyInput["cacheRetention"],
): AnthropicEphemeralCacheControl | undefined {
  const retention = resolveCacheRetention(cacheRetention);
  if (retention === "none") {
    return undefined;
  }
  // Trust explicit long-retention opt-ins for Anthropic-compatible custom providers.
  // Keep hostname gating for implicit/env-driven long retention so defaults stay conservative.
  const ttl =
    retention === "long" && (cacheRetention === "long" || isLongTtlEligibleEndpoint(baseUrl))
      ? "1h"
      : undefined;
  return { type: "ephemeral", ...(ttl ? { ttl } : {}) };
}

export function resolveAnthropicCacheOptions(
  model: Model<"anthropic-messages">,
  cacheRetention?: "short" | "long" | "none",
) {
  const supportsLongCacheRetention =
    model.compat?.supportsLongCacheRetention ?? model.provider !== "fireworks";
  const baseUrl =
    model.baseUrl?.trim() || process.env.ANTHROPIC_BASE_URL?.trim() || "https://api.anthropic.com";
  const cacheControl = resolveAnthropicEphemeralCacheControl(baseUrl, cacheRetention);
  return {
    cacheControl:
      cacheControl && !supportsLongCacheRetention ? { type: "ephemeral" as const } : cacheControl,
    supportsCacheControlOnTools:
      model.compat?.supportsCacheControlOnTools ?? model.provider !== "fireworks",
  };
}

export function buildAnthropicSystemBlocks(
  systemPrompt: string | undefined,
  isOAuthToken: boolean,
  cacheControl: AnthropicEphemeralCacheControl | undefined,
  claudeCodeVersion = ANTHROPIC_CLAUDE_CODE_VERSION,
): TextBlockParam[] | undefined {
  const blocks: TextBlockParam[] = systemPrompt
    ? [{ type: "text", text: sanitizeSurrogates(systemPrompt) }]
    : [];
  if (cacheControl) {
    applyAnthropicCacheControlToSystem(blocks, cacheControl);
  } else {
    stripAnthropicSystemPromptBoundary(blocks);
  }
  if (systemPrompt && blocks.length === 0) {
    blocks.push({ type: "text", text: "" });
  }
  if (isOAuthToken) {
    // The stable application prompt covers the OAuth preamble too; keep slots for history.
    const identityCacheControl = blocks.some((block) => block.cache_control)
      ? undefined
      : cacheControl;
    blocks.unshift(
      {
        type: "text",
        text: `x-anthropic-billing-header: cc_version=${claudeCodeVersion}; cc_entrypoint=sdk-cli;`,
      },
      {
        type: "text",
        text: "You are Claude Code, Anthropic's official CLI for Claude.",
        ...(identityCacheControl ? { cache_control: identityCacheControl } : {}),
      },
    );
  }
  return blocks.length > 0 ? blocks : undefined;
}

/** Allocate tool and history checkpoints around the prepared stable system blocks. */
export function applyAnthropicRequestCacheControl(
  payload: { system?: unknown; tools?: unknown; messages?: unknown },
  cacheControl: AnthropicEphemeralCacheControl | undefined,
  supportsCacheControlOnTools: boolean,
  cacheBreakpointOptOutMessageIndexes: ReadonlySet<number> = new Set(),
): void {
  if (!cacheControl) {
    return;
  }
  if (supportsCacheControlOnTools && Array.isArray(payload.tools)) {
    const lastTool = payload.tools.at(-1);
    if (isRecord(lastTool)) {
      lastTool.cache_control = cacheControl;
    }
  }
  const usedMarkers =
    countAnthropicCacheControlMarkers(payload.system) +
    countAnthropicCacheControlMarkers(payload.tools);
  applyAnthropicCacheControlToMessages(
    payload.messages,
    cacheControl,
    ANTHROPIC_CACHE_CONTROL_LIMIT - usedMarkers,
    cacheBreakpointOptOutMessageIndexes,
  );
}

function applyAnthropicCacheControlToSystem(
  system: unknown,
  cacheControl: AnthropicEphemeralCacheControl,
): void {
  if (!Array.isArray(system)) {
    return;
  }

  const normalizedBlocks: Array<unknown> = [];
  for (const block of system) {
    if (!block || typeof block !== "object") {
      normalizedBlocks.push(block);
      continue;
    }
    const record = block as Record<string, unknown>;
    const blockText = record.text;
    if (record.type !== "text" || typeof blockText !== "string") {
      normalizedBlocks.push(block);
      continue;
    }
    // This transport relocates nothing, so the relocatable marker must not
    // survive into the payload; the cache boundary stays for the breakpoint.
    const text = stripSystemPromptRelocatableBoundary(blockText);
    record.text = text;
    const split = splitSystemPromptCacheBoundary(text);
    if (!split) {
      if (record.cache_control === undefined) {
        record.cache_control = cacheControl;
      }
      normalizedBlocks.push(record);
      continue;
    }

    const { cache_control: existingCacheControl, ...rest } = record;
    if (split.stablePrefix) {
      normalizedBlocks.push({
        ...rest,
        text: split.stablePrefix,
        cache_control: existingCacheControl ?? cacheControl,
      });
    }
    if (split.dynamicSuffix) {
      normalizedBlocks.push({
        ...rest,
        text: split.dynamicSuffix,
      });
    }
  }

  system.splice(0, system.length, ...normalizedBlocks);
}

function stripAnthropicSystemPromptBoundary(system: unknown): void {
  if (!Array.isArray(system)) {
    return;
  }

  for (const block of system) {
    if (!block || typeof block !== "object") {
      continue;
    }
    const record = block as Record<string, unknown>;
    if (record.type === "text" && typeof record.text === "string") {
      record.text = stripSystemPromptCacheBoundary(record.text);
    }
  }
}

function isUserMessageRecord(message: unknown): message is Record<string, unknown> {
  return isRecord(message) && message.role === "user";
}

/** A user message starts a turn when it carries text or an image, not only tool results. */
function carriesUserTurnContent(message: Record<string, unknown>): boolean {
  const { content } = message;
  if (typeof content === "string") {
    return content.length > 0;
  }
  return (
    Array.isArray(content) &&
    content.some((block) => isRecord(block) && (block.type === "text" || block.type === "image"))
  );
}

/**
 * Last block of the assistant message at `index` that may carry a breakpoint. Only non-empty
 * text and tool_use qualify; thinking and server-side compaction blocks never take cache_control.
 */
function findAssistantCacheableTail(
  messages: ReadonlyArray<unknown>,
  index: number,
): Record<string, unknown> | undefined {
  const message = messages[index];
  if (!isRecord(message) || message.role !== "assistant" || !Array.isArray(message.content)) {
    return undefined;
  }
  for (let i = message.content.length - 1; i >= 0; i--) {
    const block = message.content[i];
    if (
      isRecord(block) &&
      (block.type === "tool_use" ||
        (block.type === "text" && typeof block.text === "string" && block.text.length > 0))
    ) {
      return block;
    }
  }
  return undefined;
}

/**
 * Stable history boundaries, newest first: the end of the assistant message before the newest
 * user turn (written now) and before the turn preceding it (the boundary the previous request
 * wrote). A turn is a run of consecutive user messages, so a runtime-context carrier belongs to
 * the turn it follows. Everything before the newest turn replays byte-stable, while the newest
 * turn's own messages are re-rendered between requests. Opted-out messages (ephemeral
 * context appended to the newest request only) never start a turn.
 */
function findStableHistoryBoundaries(
  messages: ReadonlyArray<unknown>,
  cacheBreakpointOptOutMessageIndexes: ReadonlySet<number>,
): Record<string, unknown>[] {
  const boundaries: Record<string, unknown>[] = [];
  let turns = 0;
  let i = messages.length - 1;
  while (i >= 0 && turns < 2) {
    if (!isUserMessageRecord(messages[i])) {
      i--;
      continue;
    }
    let startsTurn = false;
    for (; i >= 0; i--) {
      const message = messages[i];
      if (!isUserMessageRecord(message)) {
        break;
      }
      startsTurn ||= !cacheBreakpointOptOutMessageIndexes.has(i) && carriesUserTurnContent(message);
    }
    if (startsTurn) {
      turns++;
      const boundary = findAssistantCacheableTail(messages, i);
      if (boundary) {
        boundaries.push(boundary);
      }
    }
  }
  return boundaries;
}

/**
 * Allocate message breakpoints in priority order: the advancing tool-result checkpoint, then
 * the stable history boundaries, then the newest user block. Each request writes the entry the
 * next request reads at its previous boundary, so history stays cached even though the newest
 * turn's tail does not replay identically.
 */
function applyAnthropicCacheControlToMessages(
  messages: unknown,
  cacheControl: AnthropicEphemeralCacheControl,
  markerLimit: number,
  cacheBreakpointOptOutMessageIndexes: ReadonlySet<number>,
): void {
  if (!Array.isArray(messages) || messages.length === 0 || markerLimit <= 0) {
    return;
  }

  let toolResult: Record<string, unknown> | undefined;
  let newestUserBlock: Record<string, unknown> | undefined;
  let newestUserString: Record<string, unknown> | undefined;

  for (let i = messages.length - 1; i >= 0 && !newestUserBlock && !newestUserString; i--) {
    const record = messages[i];
    if (!isUserMessageRecord(record) || cacheBreakpointOptOutMessageIndexes.has(i)) {
      continue;
    }
    const content = record.content;
    if (typeof content === "string") {
      newestUserString = record;
      continue;
    }
    if (!Array.isArray(content)) {
      continue;
    }
    for (let j = content.length - 1; j >= 0 && !newestUserBlock; j--) {
      const block = content[j];
      if (!isRecord(block)) {
        continue;
      }
      if (block.type === "text" || block.type === "image") {
        newestUserBlock = block;
      } else if (block.type === "tool_result" && toolResult === undefined) {
        toolResult = block;
      }
    }
  }

  const targets = [
    toolResult,
    ...findStableHistoryBoundaries(messages, cacheBreakpointOptOutMessageIndexes),
    newestUserBlock,
  ].filter((block) => block !== undefined);
  const selected = targets.slice(0, markerLimit);
  for (const block of selected) {
    block.cache_control = cacheControl;
  }
  if (newestUserString && selected.length < markerLimit) {
    newestUserString.content = [
      { type: "text", text: newestUserString.content, cache_control: cacheControl },
    ];
  }
}

function countAnthropicCacheControlMarkers(blocks: unknown): number {
  if (!Array.isArray(blocks)) {
    return 0;
  }

  let count = 0;
  for (const block of blocks) {
    if (block && typeof block === "object" && "cache_control" in block) {
      count += 1;
    }
  }
  return count;
}

/** @deprecated Anthropic-family provider payload helper; do not use from third-party plugins. */
export function resolveAnthropicPayloadPolicy(input: AnthropicPayloadPolicyInput, model?: Model) {
  const capabilities = resolveProviderRequestCapabilities(
    {
      provider: input.provider,
      api: input.api,
      baseUrl: input.baseUrl,
      capability: "llm",
      transport: "stream",
    },
    model,
  );
  const serverCompactionPlan = resolveAnthropicServerCompactionPlan(input, input.extraParams);

  return {
    allowsServiceTier: capabilities.allowsAnthropicServiceTier,
    cacheControl:
      input.enableCacheControl === true
        ? resolveAnthropicEphemeralCacheControl(input.baseUrl, input.cacheRetention)
        : undefined,
    compactThreshold:
      serverCompactionPlan.threshold ??
      resolveAnthropicCompactThreshold(
        input.contextWindow,
        input.extraParams?.anthropicCompactThreshold,
      ),
    serviceTier: input.serviceTier,
    useServerCompaction: input.enableServerCompaction === true && serverCompactionPlan.enabled,
    ...(input.cacheTtlPruning &&
    normalizeOptionalLowercaseString(input.api) === "anthropic-messages" &&
    isDirectAnthropicModel(input)
      ? {
          toolClearing: {
            trigger: Math.max(
              50_000,
              Math.floor((parsePositiveInteger(input.contextWindow) ?? 0) * 0.3),
            ),
            clearAtLeast: Math.max(
              12_500,
              Math.floor((parsePositiveInteger(input.contextWindow) ?? 0) * 0.05),
            ),
            tools: input.cacheTtlPruning.tools,
          },
        }
      : {}),
  };
}

type AnthropicContextManagementPayload = {
  context_management?: unknown;
  tools?: unknown;
  messages?: unknown;
};

function resolveToolClearingExclusions(
  payload: AnthropicContextManagementPayload,
  filter: NonNullable<AnthropicPayloadPolicy["toolClearing"]>["tools"],
): string[] {
  const compile = (patterns: string[] | undefined) =>
    (patterns ?? [])
      .map((pattern) => pattern.trim().toLowerCase())
      .filter(Boolean)
      .map(
        (pattern) =>
          new RegExp(`^${pattern.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replaceAll("\\*", ".*")}$`),
      );
  const allow = compile(filter?.allow);
  const deny = compile(filter?.deny);
  if (allow.length === 0 && deny.length === 0) {
    return [];
  }
  const excluded = new Set(
    filter?.deny?.map((name) => name.trim()).filter((name) => name && !name.includes("*")),
  );
  const candidates = new Set<string>();
  for (const tool of Array.isArray(payload.tools) ? payload.tools : []) {
    if (isRecord(tool) && typeof tool.name === "string") {
      candidates.add(tool.name);
    }
  }
  // Pruning filters apply to history too: removing a tool must not make its results clearable.
  for (const message of Array.isArray(payload.messages) ? payload.messages : []) {
    if (!isRecord(message) || !Array.isArray(message.content)) {
      continue;
    }
    for (const block of message.content) {
      if (isRecord(block) && block.type === "tool_use" && typeof block.name === "string") {
        candidates.add(block.name);
      }
    }
  }
  for (const toolName of candidates) {
    const name = toolName.trim().toLowerCase();
    if (
      deny.some((pattern) => pattern.test(name)) ||
      (allow.length > 0 && !allow.some((pattern) => pattern.test(name)))
    ) {
      excluded.add(toolName);
    }
  }
  return [...excluded].toSorted();
}

function applyAnthropicContextManagementEdits(
  payload: AnthropicContextManagementPayload,
  policy: AnthropicPayloadPolicy,
): void {
  if (payload.context_management !== undefined) {
    return;
  }
  const edits: NonNullable<BetaContextManagementConfig["edits"]> = [];
  if (policy.toolClearing) {
    edits.push({
      type: "clear_tool_uses_20250919",
      trigger: { type: "input_tokens", value: policy.toolClearing.trigger },
      keep: { type: "tool_uses", value: 3 },
      clear_at_least: { type: "input_tokens", value: policy.toolClearing.clearAtLeast },
      exclude_tools: resolveToolClearingExclusions(payload, policy.toolClearing.tools),
      clear_tool_inputs: false,
    });
  }
  // Clear cheap-to-discard tool results before asking the server to summarize the remaining context.
  if (policy.useServerCompaction) {
    edits.push({
      type: "compact_20260112",
      trigger: { type: "input_tokens", value: policy.compactThreshold },
    });
  }
  if (edits.length > 0) {
    payload.context_management = { edits };
  }
}

export function applyAnthropicContextManagementToRequest(
  payload: AnthropicContextManagementPayload,
  model: Model,
  options: AnthropicContextManagementOptions | undefined,
  directApiKeyBetaHeader: string | undefined,
): void {
  if (
    directApiKeyBetaHeader === undefined ||
    (!options?.cacheTtlPruning && !options?.anthropicServerCompaction)
  ) {
    return;
  }
  applyAnthropicContextManagementEdits(
    payload,
    resolveAnthropicPayloadPolicy({
      ...model,
      // This adapter owns the wire API; simple-dispatch aliases remain on the replay identity.
      api: "anthropic-messages",
      enableServerCompaction: true,
      extraParams: { ...options },
      cacheTtlPruning: options?.cacheTtlPruning,
    }),
  );
}

export function resolveAnthropicRequestBetaHeader(
  payload: AnthropicContextManagementPayload & { fallbacks?: unknown },
  directApiKeyBetaHeader: string | undefined,
): string | undefined {
  if (directApiKeyBetaHeader === undefined) {
    return directApiKeyBetaHeader;
  }
  const edits = isRecord(payload.context_management) ? payload.context_management.edits : undefined;
  const betas = new Set(
    directApiKeyBetaHeader
      .split(",")
      .map((beta) => beta.trim())
      .filter(Boolean),
  );
  // Payload-required betas must survive model and per-request header overrides.
  if (payload.fallbacks === ANTHROPIC_SERVER_SIDE_FALLBACKS) {
    betas.add(ANTHROPIC_SERVER_SIDE_FALLBACK_BETA);
  }
  if (
    Array.isArray(payload.messages) &&
    payload.messages.some(
      (message) =>
        isRecord(message) && message.role === "system" && message.clear_at === "next_user_message",
    )
  ) {
    betas.add("mid-conversation-system-clear-at-2026-08-21");
  }
  for (const edit of Array.isArray(edits) ? edits : []) {
    if (!isRecord(edit)) {
      continue;
    }
    if (edit.type === "clear_tool_uses_20250919") {
      betas.add("context-management-2025-06-27");
    } else if (edit.type === "compact_20260112") {
      betas.add("compact-2026-01-12");
    }
  }
  return [...betas].join(",");
}

export function logAnthropicContextEdits(event: unknown): void {
  if (!isRecord(event) || !isRecord(event.context_management)) {
    return;
  }
  const edits = event.context_management.applied_edits;
  let clearedTools = 0;
  let clearedTokens = 0;
  for (const edit of Array.isArray(edits) ? edits : []) {
    if (
      !isRecord(edit) ||
      edit.type !== "clear_tool_uses_20250919" ||
      typeof edit.cleared_tool_uses !== "number" ||
      !Number.isSafeInteger(edit.cleared_tool_uses) ||
      edit.cleared_tool_uses < 0 ||
      typeof edit.cleared_input_tokens !== "number" ||
      !Number.isSafeInteger(edit.cleared_input_tokens) ||
      edit.cleared_input_tokens < 0
    ) {
      continue;
    }
    clearedTools = Math.min(Number.MAX_SAFE_INTEGER, clearedTools + edit.cleared_tool_uses);
    clearedTokens = Math.min(Number.MAX_SAFE_INTEGER, clearedTokens + edit.cleared_input_tokens);
  }
  if (clearedTools > 0 || clearedTokens > 0) {
    getAiTransportHost().logInfo(
      "anthropic",
      `server-side context edit: cleared ${clearedTools} tool results (${clearedTokens} input tokens)`,
    );
  }
}

/** @deprecated Anthropic-family provider payload helper; do not use from third-party plugins. */
export function applyAnthropicPayloadPolicyToParams(
  payloadObj: Record<string, unknown>,
  policy: AnthropicPayloadPolicy,
  cacheBreakpointOptOutMessageIndexes: ReadonlySet<number>,
): void {
  if (
    policy.allowsServiceTier &&
    policy.serviceTier !== undefined &&
    payloadObj.service_tier === undefined
  ) {
    payloadObj.service_tier = policy.serviceTier;
  }

  if (policy.cacheControl) {
    applyAnthropicCacheControlToSystem(payloadObj.system, policy.cacheControl);
  } else {
    stripAnthropicSystemPromptBoundary(payloadObj.system);
  }

  applyAnthropicContextManagementEdits(payloadObj, policy);

  applyAnthropicRequestCacheControl(
    payloadObj,
    policy.cacheControl,
    false,
    cacheBreakpointOptOutMessageIndexes,
  );
}

/** @deprecated Anthropic-family provider payload helper; do not use from third-party plugins. */
export function applyAnthropicEphemeralCacheControlMarkers(
  payloadObj: Record<string, unknown>,
  cacheControl: AnthropicEphemeralCacheControl | null = { type: "ephemeral" },
): void {
  const messages = payloadObj.messages;
  if (!Array.isArray(messages)) {
    return;
  }

  for (const message of messages as Array<{ role?: string; content?: unknown }>) {
    if (message.role === "system" || message.role === "developer") {
      if (!cacheControl) {
        continue;
      }
      if (typeof message.content === "string") {
        message.content = [{ type: "text", text: message.content, cache_control: cacheControl }];
        continue;
      }
      if (Array.isArray(message.content) && message.content.length > 0) {
        const last = message.content[message.content.length - 1];
        if (last && typeof last === "object") {
          const record = last as Record<string, unknown>;
          if (record.type !== "thinking" && record.type !== "redacted_thinking") {
            record.cache_control = cacheControl;
          }
        }
      }
      continue;
    }

    if (message.role === "assistant" && Array.isArray(message.content)) {
      for (const block of message.content) {
        if (!block || typeof block !== "object") {
          continue;
        }
        const record = block as Record<string, unknown>;
        if (record.type === "thinking" || record.type === "redacted_thinking") {
          delete record.cache_control;
        }
      }
    }
  }
}
