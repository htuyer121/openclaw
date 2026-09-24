import { describe, expect, it, vi } from "vitest";
import { createEmptyPluginRegistry } from "../plugins/registry-empty.js";
import {
  commitWorkerDiscovery,
  releaseWorkerDiscoveries,
  releaseWorkerDiscovery,
  workerDiscoveries,
  type WorkerDiscovery,
  type WorkerGeneration,
} from "./prepared-model-catalog-worker.generations.js";

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

describe("worker discovery retention", () => {
  it("keeps four exact scopes and retires the least recently successful one", async () => {
    const generation: Pick<WorkerGeneration, "discoveries"> = {};
    const entries = ["a", "b", "c", "d", "e"].map(discovery);
    for (const entry of entries.slice(0, 4)) {
      await commitWorkerDiscovery(generation, entry);
    }
    const cache = workerDiscoveries(generation);
    // An attempted read alone must not protect a failed request from eviction.
    expect(cache.get("b")).toBe(entries[1]);
    await commitWorkerDiscovery(generation, entries[0]!);
    const releaseB = entries[1]!.release;
    await commitWorkerDiscovery(generation, entries[4]!);
    expect([...cache.keys()]).toEqual(["c", "d", "a", "e"]);
    expect(releaseB).toHaveBeenCalledTimes(1);
    for (const index of [0, 2, 3, 4]) {
      expect(entries[index]!.release).not.toHaveBeenCalled();
    }
    await releaseWorkerDiscoveries(generation);
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
    const generation = {};
    const entries = ["a", "b", "c"].map(discovery);
    const releases = entries.map((entry) => entry.release);
    releases[0]!.mockImplementation(async () => {
      expect(workerDiscoveries(generation).size).toBe(0);
      throw new Error("retained native cleanup failure");
    });
    for (const entry of entries) {
      await commitWorkerDiscovery(generation, entry);
    }
    await expect(releaseWorkerDiscoveries(generation)).rejects.toThrow(
      "Catalog discovery registries failed to retire",
    );
    await releaseWorkerDiscoveries(generation);
    for (const release of releases) {
      expect(release).toHaveBeenCalledTimes(1);
    }
  });

  it("detaches a failed eviction and never retries its release", async () => {
    const generation = {};
    const entries = ["a", "b", "c", "d", "e"].map(discovery);
    const releaseA = entries[0]!.release;
    releaseA.mockRejectedValue(new Error("eviction failed"));
    for (const entry of entries.slice(0, 4)) {
      await commitWorkerDiscovery(generation, entry);
    }
    await expect(commitWorkerDiscovery(generation, entries[4]!)).rejects.toThrow("eviction failed");
    expect(workerDiscoveries(generation).has("a")).toBe(false);
    await releaseWorkerDiscovery(entries[0]!);
    await releaseWorkerDiscoveries(generation);
    expect(releaseA).toHaveBeenCalledTimes(1);
  });
});
