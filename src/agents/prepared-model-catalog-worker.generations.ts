import type { PluginRegistry } from "../plugins/registry-types.js";
import type { PreparedModelRuntimeAgentFacts } from "./prepared-model-runtime.catalog-contract.js";
import { ownPreparedPluginGeneration } from "./prepared-model-runtime.plugin-lifetime.js";
import type { PreparedModelRuntimePluginGeneration } from "./prepared-model-runtime.types.js";

export type WorkerDiscovery = {
  key: string;
  registry: PluginRegistry;
  release?: () => Promise<void>;
};

export type WorkerGeneration = {
  agentFacts: PreparedModelRuntimeAgentFacts;
  pluginGeneration: PreparedModelRuntimePluginGeneration;
  reconstructedFingerprint: string;
  discoveries?: Map<string, WorkerDiscovery>;
};

type DiscoveryOwner = Pick<WorkerGeneration, "discoveries">;

// Keep exact scopes rather than broadening a registry's admission. The bound is
// per agent generation; a larger recurring working set may still evict entries.
const MAX_DISCOVERY_REGISTRIES = 4;

export function workerDiscoveries(prepared: DiscoveryOwner) {
  return (prepared.discoveries ??= new Map<string, WorkerDiscovery>());
}

export async function releaseWorkerDiscovery(entry: WorkerDiscovery) {
  // A failed release remains in the native resource owner's custody. Detach our
  // handle before disposal so request cleanup cannot reuse or retry that release.
  const release = entry.release;
  entry.release = undefined;
  await release?.();
}

export async function releaseWorkerDiscoveries(prepared: DiscoveryOwner) {
  const cache = workerDiscoveries(prepared);
  const entries = [...cache.values()];
  cache.clear();
  const failures: unknown[] = [];
  for (const entry of entries) {
    try {
      await releaseWorkerDiscovery(entry);
    } catch (error) {
      failures.push(error);
    }
  }
  if (failures.length) {
    throw new AggregateError(failures, "Catalog discovery registries failed to retire");
  }
}

export async function commitWorkerDiscovery(prepared: DiscoveryOwner, entry: WorkerDiscovery) {
  const cache = workerDiscoveries(prepared);
  // Call only after successful request work is idle. Failed requests neither
  // publish a newly acquired registry nor promote a previous entry in the LRU.
  cache.delete(entry.key);
  cache.set(entry.key, entry);
  if (cache.size > MAX_DISCOVERY_REGISTRIES) {
    const oldest = cache.values().next().value!;
    cache.delete(oldest.key);
    await releaseWorkerDiscovery(oldest);
  }
}

// Keep custody outside the request closure, which also captures the predecessor.
export function retainWorkerGeneration(prepared: WorkerGeneration): () => Promise<void> {
  const releaseBase = ownPreparedPluginGeneration(prepared.pluginGeneration).retain();
  return async () => {
    try {
      await releaseWorkerDiscoveries(prepared);
    } finally {
      await releaseBase();
    }
  };
}
