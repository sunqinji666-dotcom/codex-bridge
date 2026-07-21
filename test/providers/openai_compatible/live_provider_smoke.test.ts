import assert from 'node:assert/strict';
import test from 'node:test';
import { loadCodexProfilesFromEnv } from '../../../src/providers/codex/config.js';
import { OpenAICompatibleResponsesAdapterServer } from '../../../src/providers/openai_compatible/responses_adapter_server.js';
import type { ProviderProfile } from '../../../src/types/provider.js';

type LiveProviderSpec = {
  profileId: string;
  name: string;
  apiKeyEnvHint: string[];
};

const LIVE_FLAG = process.env.CODEXBRIDGE_TEST_LIVE_OPENAI_COMPATIBLE === '1';
const loadedProfiles = loadCodexProfilesFromEnv(process.env);
const PROVIDERS: LiveProviderSpec[] = [{
  profileId: 'deepseek',
  name: 'DeepSeek',
  apiKeyEnvHint: ['DEEPSEEK_API_KEY'],
}, {
  profileId: 'minimax',
  name: 'MiniMax',
  apiKeyEnvHint: ['MINIMAX_API_KEY'],
}, {
  profileId: 'qwen',
  name: 'Qwen',
  apiKeyEnvHint: ['QWEN_API_KEY', 'DASHSCOPE_API_KEY'],
}, {
  profileId: 'openrouter',
  name: 'OpenRouter',
  apiKeyEnvHint: ['OPENROUTER_API_KEY'],
}];

for (const provider of PROVIDERS) {
  const resolved = resolveProviderFromCodexBridgeProfile(provider);
  test(`live OpenAI-compatible adapter smoke: ${provider.name}`, {
    skip: skipReason(provider, resolved),
    timeout: 90_000,
  }, async () => {
    assert.ok(resolved.profile);
    const server = new OpenAICompatibleResponsesAdapterServer({
      apiKey: resolved.apiKey,
      upstreamBaseUrl: resolved.baseUrl,
      defaultModel: resolved.model,
      models: resolved.models,
      providerName: provider.name,
      providerKind: resolved.profile.providerKind,
      fetchImpl: ((input, init) => fetch(input, {
        ...init,
        signal: AbortSignal.timeout(60_000),
      })) as typeof fetch,
      providerCapabilities: resolved.capabilities,
      upstreamChatCompletionsPath: resolved.upstreamChatCompletionsPath,
      ownedBy: resolved.ownedBy,
    });
    await server.start();
    try {
      const modelsResponse = await fetch(`${server.baseUrl}/v1/models`, {
        signal: AbortSignal.timeout(10_000),
      });
      const modelsBody = await modelsResponse.json() as any;
      assert.equal(modelsResponse.status, 200);
      assert.equal(
        normalizeArray(modelsBody?.data).some((model) => normalizeString(model?.id) === resolved.model),
        true,
        `${provider.name} profile model catalog did not expose ${resolved.model}`,
      );

      const response = await fetch(`${server.baseUrl}/v1/responses`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: resolved.model,
          instructions: 'Return only the final answer. Do not explain.',
          input: 'Final answer must be exactly: OK',
          max_output_tokens: 128,
          stream: false,
        }),
        signal: AbortSignal.timeout(70_000),
      });
      const text = await response.text();
      assert.equal(response.status, 200, `${provider.name} status ${response.status}: ${text.slice(0, 1000)}`);
      const body = JSON.parse(text);
      const outputText = collectResponseOutputText(body);
      assert.match(outputText, /\bOK\b/i, `${provider.name} output did not contain OK: ${outputText}`);
    } finally {
      await server.stop();
    }
  });
}

function skipReason(provider: LiveProviderSpec, resolved: ResolvedProvider): string | false {
  if (!LIVE_FLAG) {
    return 'set CODEXBRIDGE_TEST_LIVE_OPENAI_COMPATIBLE=1 to run live provider smoke tests';
  }
  if (!resolved.profile) {
    return `missing CodexBridge provider profile: ${provider.profileId}; set ${provider.apiKeyEnvHint.join(' or ')}`;
  }
  if (!resolved.apiKey) {
    return `missing API key env: ${resolved.apiKeyEnv || provider.apiKeyEnvHint.join(' or ')}`;
  }
  return false;
}

type ResolvedProvider = {
  profile: ProviderProfile | null;
  apiKeyEnv: string;
  apiKey: string;
  baseUrl: string;
  model: string;
  models: any[];
  capabilities: any;
  upstreamChatCompletionsPath: string | null;
  ownedBy: string | null;
};

function resolveProviderFromCodexBridgeProfile(provider: LiveProviderSpec): ResolvedProvider {
  const profile = loadedProfiles.profiles.find((entry) => entry.id === provider.profileId) ?? null;
  const config = profile?.config as Record<string, any> | undefined;
  const apiKeyEnv = normalizeString(config?.apiKeyEnv);
  const model = normalizeString(config?.defaultModel);
  return {
    profile,
    apiKeyEnv,
    apiKey: apiKeyEnv ? normalizeString(process.env[apiKeyEnv]) : '',
    baseUrl: normalizeString(config?.baseUrl),
    model,
    models: normalizeArray(config?.modelCatalog).length > 0 ? normalizeArray(config?.modelCatalog) : [{ id: model }],
    capabilities: config?.capabilities ?? null,
    upstreamChatCompletionsPath: normalizeString(config?.upstreamChatCompletionsPath) || null,
    ownedBy: normalizeString(config?.ownedBy) || null,
  };
}

function collectResponseOutputText(response: any): string {
  const parts: string[] = [];
  for (const item of Array.isArray(response?.output) ? response.output : []) {
    for (const content of Array.isArray(item?.content) ? item.content : []) {
      if (typeof content?.text === 'string') {
        parts.push(content.text);
      }
    }
  }
  return parts.join('\n').trim();
}

function normalizeString(value: unknown): string {
  return typeof value === 'string' ? value.trim() : '';
}

function normalizeArray(value: unknown): any[] {
  return Array.isArray(value) ? value : [];
}
