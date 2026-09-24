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

export type WorkerGenerationOwner = {
  fingerprint: string;
  prepared: WorkerGeneration;
  release: () => Promise<void>;
};

// Bound retained registrations across the whole worker, including base generations.
// This is a settled cache bound, not a byte limit on plugin code or native ESM jobs.
export const MAX_CATALOG_WORKER_REGISTRIES = 32;
const successfulUses = new WeakMap<WorkerGeneration | WorkerDiscovery, number>();
let nextSuccessfulUse = 0;

function recordSuccessfulUse(entry: WorkerGeneration | WorkerDiscovery) {
  successfulUses.set(entry, ++nextSuccessfulUse);
}

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

function detachWorkerDiscoveries(prepared: DiscoveryOwner) {
  const cache = workerDiscoveries(prepared);
  const entries = [...cache.values()];
  cache.clear();
  return entries;
}

async function releaseWorkerRegistries(releases: Array<() => Promise<void>>, message: string) {
  const failures: unknown[] = [];
  for (const release of releases) {
    try {
      await release();
    } catch (error) {
      failures.push(error);
    }
  }
  if (failures.length) {
    throw new AggregateError(failures, message);
  }
}

export function releaseWorkerDiscoveries(prepared: DiscoveryOwner) {
  return releaseWorkerRegistries(
    detachWorkerDiscoveries(prepared).map((entry) => () => releaseWorkerDiscovery(entry)),
    "Catalog discovery registries failed to retire",
  );
}

export function commitWorkerDiscovery(prepared: DiscoveryOwner, entry: WorkerDiscovery) {
  const cache = workerDiscoveries(prepared);
  // Call only after successful request work is idle. Failed requests neither
  // publish a newly acquired registry nor promote a previous entry in the LRU.
  cache.delete(entry.key);
  cache.set(entry.key, entry);
  recordSuccessfulUse(entry);
}

function detachWorkerGeneration(
  owner: WorkerGenerationOwner,
  releases: Array<() => Promise<void>>,
) {
  for (const entry of detachWorkerDiscoveries(owner.prepared)) {
    releases.push(() => releaseWorkerDiscovery(entry));
  }
  // Keep the base's metadata custody until every detached child cleanup was attempted.
  releases.push(owner.release);
}

/** One owner for successful base slots and their worker-wide registry retention budget. */
export class WorkerGenerationCache {
  private readonly slots = new Map<string, WorkerGenerationOwner>();

  constructor(private readonly maxRegistries = MAX_CATALOG_WORKER_REGISTRIES) {
    if (!Number.isSafeInteger(maxRegistries) || maxRegistries < 1) {
      throw new RangeError("Catalog worker registry budget must be a positive integer");
    }
  }

  get(key: string) {
    return this.slots.get(key);
  }

  /** Takes custody before disposal; call only after the serial request has drained. */
  commit(key: string, owner: WorkerGenerationOwner): Promise<void> {
    const previous = this.slots.get(key);
    this.slots.set(key, owner);
    recordSuccessfulUse(owner.prepared);
    const releases: Array<() => Promise<void>> = [];
    if (previous && previous.prepared !== owner.prepared) {
      // Publish the successor before retiring shared source registrations.
      detachWorkerGeneration(previous, releases);
    }
    const candidates: Array<{
      key: string;
      owner: WorkerGenerationOwner;
      discovery?: WorkerDiscovery;
    }> = [];
    for (const [slotKey, retained] of this.slots) {
      candidates.push({ key: slotKey, owner: retained });
      for (const discovery of workerDiscoveries(retained.prepared).values()) {
        candidates.push({ key: slotKey, owner: retained, discovery });
      }
    }
    candidates.sort(
      (left, right) =>
        (successfulUses.get(left.discovery ?? left.owner.prepared) ?? 0) -
        (successfulUses.get(right.discovery ?? right.owner.prepared) ?? 0),
    );
    let count = candidates.length;
    for (const candidate of candidates) {
      if (count <= this.maxRegistries) {
        break;
      }
      if (this.slots.get(candidate.key) !== candidate.owner) {
        continue;
      }
      const discoveries = workerDiscoveries(candidate.owner.prepared);
      if (candidate.discovery) {
        const entry = candidate.discovery;
        if (discoveries.get(entry.key) === entry) {
          discoveries.delete(entry.key);
          count -= 1;
          releases.push(() => releaseWorkerDiscovery(entry));
        }
      } else {
        this.slots.delete(candidate.key);
        count -= 1 + discoveries.size;
        detachWorkerGeneration(candidate.owner, releases);
      }
    }
    // Detach every victim before awaiting any release, including when an earlier release fails.
    return releaseWorkerRegistries(releases, "Catalog worker registries failed to retire");
  }

  clear(): Promise<void> {
    const owners = [...this.slots.values()];
    this.slots.clear();
    const releases: Array<() => Promise<void>> = [];
    for (const owner of owners) {
      detachWorkerGeneration(owner, releases);
    }
    return releaseWorkerRegistries(releases, "Catalog agent generations failed to retire");
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
