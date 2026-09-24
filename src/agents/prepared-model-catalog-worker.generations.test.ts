import { describe, expect, it, vi } from "vitest";
import { createPluginMetadataSnapshotFixture } from "../plugins/plugin-metadata.test-support.js";
import { createEmptyPluginRegistry } from "../plugins/registry-empty.js";
import {
  commitWorkerDiscovery,
  releaseWorkerDiscoveries,
  releaseWorkerDiscovery,
  workerDiscoveries,
  WorkerGenerationCache,
  type WorkerDiscovery,
  type WorkerGeneration,
} from "./prepared-model-catalog-worker.generations.js";
import { AuthStorage } from "./sessions/auth-storage.js";

vi.mock("./prepared-model-runtime.plugin-lifetime.js", () => ({
  ownPreparedPluginGeneration: vi.fn(),
}));

function discovery(key: string) {
  return {
    key,
    registry: createEmptyPluginRegistry(),
    release: vi.fn(async () => {}),
  } satisfies WorkerDiscovery;
}

function generation(key: string) {
  const prepared: WorkerGeneration = {
    reconstructedFingerprint: key,
    pluginGeneration: {
      pluginMetadataSnapshot: createPluginMetadataSnapshotFixture(),
      inlineProviderModels: [],
      configuredCatalogEntries: [],
    },
    agentFacts: {
      input: { agentDir: key, config: {} },
      env: {},
      authStore: { version: 1, profiles: {} },
      templateAuthStorage: AuthStorage.inMemory({}),
      credentials: {},
      providerIds: [],
      configuredModelRefs: [],
      configuredRuntimeModels: [],
      runtimeCapabilityModels: [],
      configuredGeneratedCatalogPluginIds: [],
    },
  };
  return {
    fingerprint: key,
    prepared,
    release: vi.fn(async () => releaseWorkerDiscoveries(prepared)),
  };
}

describe("worker discovery retention", () => {
  it("reuses more than four exact scopes within the worker-wide budget", async () => {
    const cache = new WorkerGenerationCache(7);
    const owner = generation("alpha");
    const entries = ["a", "b", "c", "d", "e", "f"].map(discovery);
    for (const entry of entries) {
      await commitWorkerDiscovery(owner.prepared, entry);
      await cache.commit("alpha", owner);
    }
    expect([...workerDiscoveries(owner.prepared).keys()]).toEqual(entries.map(({ key }) => key));
    for (const entry of entries) {
      expect(entry.release).not.toHaveBeenCalled();
    }
    expect(owner.release).not.toHaveBeenCalled();
    await cache.clear();
  });

  it("evicts the oldest successful discovery across agents without promoting attempted reads", async () => {
    const cache = new WorkerGenerationCache(5);
    const alpha = generation("alpha");
    const beta = generation("beta");
    const [a, b, c, d] = ["a", "b", "c", "d"].map(discovery);
    for (const entry of [a!, b!]) {
      await commitWorkerDiscovery(alpha.prepared, entry);
      await cache.commit("alpha", alpha);
    }
    await commitWorkerDiscovery(beta.prepared, c!);
    await cache.commit("beta", beta);
    await commitWorkerDiscovery(alpha.prepared, a!);
    await cache.commit("alpha", alpha);
    expect(workerDiscoveries(alpha.prepared).get("b")).toBe(b);
    expect(cache.get("beta")).toBe(beta);
    const releaseB = b!.release;
    await commitWorkerDiscovery(beta.prepared, d!);
    await cache.commit("beta", beta);
    expect(releaseB).toHaveBeenCalledTimes(1);
    expect([...workerDiscoveries(alpha.prepared).keys()]).toEqual(["a"]);
    expect([...workerDiscoveries(beta.prepared).keys()]).toEqual(["c", "d"]);
    expect(alpha.release).not.toHaveBeenCalled();
    expect(beta.release).not.toHaveBeenCalled();
    await cache.clear();
  });

  it("keeps identical scopes isolated between agent generations", async () => {
    const first = {};
    const second = {};
    const a = discovery("same-exact-scope");
    const b = discovery("same-exact-scope");
    await commitWorkerDiscovery(first, a);
    await commitWorkerDiscovery(second, b);
    await releaseWorkerDiscoveries(first);
    expect(workerDiscoveries(first).size).toBe(0);
    expect(workerDiscoveries(second).get(b.key)?.registry).toBe(b.registry);
    expect(b.release).not.toHaveBeenCalled();
    await releaseWorkerDiscoveries(second);
  });

  it("detaches every entry and drains other releases after a retirement failure", async () => {
    const prepared = {};
    const entries = ["a", "b", "c"].map(discovery);
    const releases = entries.map((entry) => entry.release);
    releases[0]!.mockImplementation(async () => {
      expect(workerDiscoveries(prepared).size).toBe(0);
      throw new Error("retained native cleanup failure");
    });
    for (const entry of entries) {
      await commitWorkerDiscovery(prepared, entry);
    }
    await expect(releaseWorkerDiscoveries(prepared)).rejects.toThrow(
      "Catalog discovery registries failed to retire",
    );
    await releaseWorkerDiscoveries(prepared);
    for (const release of releases) {
      expect(release).toHaveBeenCalledTimes(1);
    }
  });

  it("detaches every eviction before cleanup and drains the base after a child release fails", async () => {
    const cache = new WorkerGenerationCache(2);
    const alpha = generation("alpha");
    const beta = generation("beta");
    const a = discovery("a");
    const b = discovery("b");
    const releaseA = a.release;
    releaseA.mockImplementation(async () => {
      expect(cache.get("alpha")).toBeUndefined();
      expect(workerDiscoveries(alpha.prepared).size).toBe(0);
      expect(cache.get("beta")).toBe(beta);
      throw new Error("eviction failed");
    });
    await commitWorkerDiscovery(alpha.prepared, a);
    await cache.commit("alpha", alpha);
    await commitWorkerDiscovery(beta.prepared, b);
    await expect(cache.commit("beta", beta)).rejects.toThrow(
      "Catalog worker registries failed to retire",
    );
    expect(alpha.release).toHaveBeenCalledTimes(1);
    expect(beta.release).not.toHaveBeenCalled();
    await releaseWorkerDiscovery(a);
    await cache.clear();
    expect(releaseA).toHaveBeenCalledTimes(1);
  });

  it("publishes a replacement before releasing its predecessor and preserves the successor on failure", async () => {
    const cache = new WorkerGenerationCache(2);
    const old = generation("old");
    const next = generation("next");
    await cache.commit("alpha", old);
    old.release.mockImplementation(async () => {
      expect(cache.get("alpha")).toBe(next);
      throw new Error("replacement release failed");
    });
    await expect(cache.commit("alpha", next)).rejects.toThrow(
      "Catalog worker registries failed to retire",
    );
    await cache.commit("alpha", next);
    expect(old.release).toHaveBeenCalledTimes(1);
    expect(next.release).not.toHaveBeenCalled();
    await cache.clear();
  });
});
