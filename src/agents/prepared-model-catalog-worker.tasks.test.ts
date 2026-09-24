import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { OpenClawConfig } from "../config/types.openclaw.js";
import { createPluginMetadataSnapshotFixture } from "../plugins/plugin-metadata.test-support.js";
import * as agentScope from "./agent-scope-config.js";
import type {
  PreparedModelCatalogWorkerData,
  PreparedModelCatalogWorkerInput,
  PreparedModelWorkerRequest,
} from "./prepared-model-catalog-worker.js";
import { createPreparedCatalogTaskHandler } from "./prepared-model-catalog.worker.js";
import type { PreparedModelRuntimeAgentFacts } from "./prepared-model-runtime.catalog-contract.js";
import type {
  PreparedModelRuntimeInput,
  PreparedModelRuntimePluginGeneration,
} from "./prepared-model-runtime.types.js";
import { AuthStorage } from "./sessions/auth-storage.js";

const mocks = vi.hoisted(() => ({
  prepare: vi.fn(),
  discard: vi.fn(),
  refreshAuth: vi.fn(),
  releases: new Map<object, ReturnType<typeof vi.fn<() => Promise<void>>>>(),
}));

// Keep the actual task handler and request cleanup, while replacing construction
// and IO boundaries. These tests do not attach to Vitest's worker message port.
vi.mock("../infra/worker-task-server.js", () => ({ serveWorkerTasks: vi.fn() }));
vi.mock("../plugins/plugin-package-metadata-capture.js", () => ({
  withPluginSourceCaptureDirectory: (_directory: string, run: () => unknown) => run(),
}));
vi.mock("../config/runtime-snapshot.js", () => ({ setRuntimeConfigSnapshot: vi.fn() }));
vi.mock("./prepared-model-runtime.facts.js", () => ({
  prepareWorkspaceBuildGroup: mocks.prepare,
  fingerprintPreparedRuntimeFacts: (value: unknown) => JSON.stringify(value),
}));
vi.mock("./prepared-model-catalog-worker.js", () => ({
  PREPARED_MODEL_CATALOG_WORKER_TIMEOUT_MS: 180_000,
  fingerprintPreparedModelCatalogGeneration: (value: {
    input: { agentDir: string };
    providerIds: readonly string[];
  }) => `${value.input.agentDir}:${value.providerIds.join(",")}`,
  fingerprintPreparedModelWorkerRequest: (value: { generationFingerprint: string }) =>
    value.generationFingerprint,
}));
vi.mock("./prepared-model-runtime.plugin-lifetime.js", () => ({
  discardPreparedPluginGeneration: mocks.discard,
  retainPreparedPluginRegistry: vi.fn(),
  ownPreparedPluginGeneration: (generation: object) => ({
    retain: () => {
      const release = mocks.releases.get(generation);
      if (!release) {
        throw new Error("Missing synthetic generation owner");
      }
      return release;
    },
  }),
}));
vi.mock("./auth-profiles/store-runtime.js", () => ({
  loadAuthProfileStoreWithoutExternalProfiles: () => ({ version: 1, profiles: {} }),
  updateAuthProfileStoreWithLock: vi.fn(() => {
    throw new Error("Unexpected durable auth mutation in catalog task fixture");
  }),
}));
vi.mock("./auth-profiles/external-auth-runtime.js", () => ({
  overlayExternalAuthProfiles: mocks.refreshAuth,
}));
vi.mock("./agent-auth-discovery.js", () => ({
  resolveAmbientAgentCredentialsForDiscovery: () => ({}),
}));

const stateDir = path.resolve("/tmp/catalog-task-fixture");
const agentDir = (agentId: string) => path.join(stateDir, "agents", agentId, "agent");
const config: OpenClawConfig = {
  agents: {
    entries: { alpha: { agentDir: agentDir("alpha") }, beta: { agentDir: agentDir("beta") } },
  },
};
const request: PreparedModelWorkerRequest = {
  kind: "auth-refresh",
  providerIds: [],
  syntheticAuth: [],
  clawInstallSchemaVersions: {
    path: path.join(stateDir, "state.sqlite"),
    snapshot: { kind: "ready", schemaVersions: [] },
  },
};

function input(
  agentId: string | undefined = "alpha",
  revision = "initial",
  directory = agentDir(agentId ?? "alpha"),
): PreparedModelCatalogWorkerInput {
  const { normalizePluginId: _normalizePluginId, ...pluginMetadataSnapshot } =
    createPluginMetadataSnapshotFixture();
  return {
    kind: "catalog",
    generationFingerprint: `${directory}:${revision}`,
    input: {
      ...(agentId ? { agentId } : {}),
      agentDir: directory,
      config,
      env: { OPENCLAW_STATE_DIR: stateDir },
    },
    sourceConfigForSecrets: config,
    configResolutionFacts: null,
    sourceConfigResolutionFacts: null,
    authStore: { version: 1, profiles: {} },
    providerIds: [revision],
    preferBuiltPluginArtifacts: false,
    pluginMetadataSnapshot,
  };
}

function gatewayHandler() {
  const run = createPreparedCatalogTaskHandler({
    kind: "gateway",
    sourceCaptureDirectory: path.join(stateDir, "captures"),
  });
  return (value: PreparedModelCatalogWorkerInput) => run({ value, request });
}

const releases = () => [...mocks.releases.values()];

beforeEach(() => {
  vi.restoreAllMocks();
  mocks.prepare.mockReset();
  mocks.discard.mockReset().mockResolvedValue(undefined);
  mocks.refreshAuth.mockReset().mockImplementation((store) => store);
  mocks.releases.clear();
  mocks.prepare.mockImplementation(async ([value]: PreparedModelRuntimeInput[]) => {
    const generation: PreparedModelRuntimePluginGeneration = {
      pluginMetadataSnapshot: createPluginMetadataSnapshotFixture(),
      inlineProviderModels: [],
      configuredCatalogEntries: [],
    };
    mocks.releases.set(generation, vi.fn<() => Promise<void>>().mockResolvedValue(undefined));
    const facts: PreparedModelRuntimeAgentFacts = {
      input: value!,
      env: value!.env ?? {},
      authStore: { version: 1, profiles: {} },
      templateAuthStorage: AuthStorage.inMemory({}),
      credentials: {},
      providerIds: [],
      configuredModelRefs: [],
      configuredRuntimeModels: [],
      runtimeCapabilityModels: [],
      configuredGeneratedCatalogPluginIds: [],
    };
    return { agentFacts: [facts], pluginGeneration: generation };
  });
});
afterEach(() => vi.restoreAllMocks());

describe("catalog worker task generation ownership", () => {
  it("retains one generation per configured agent and replaces only that agent's fingerprint", async () => {
    const run = gatewayHandler();
    for (const agentId of ["alpha", "beta", "alpha", "beta"]) {
      expect(await run(input(agentId))).toMatchObject({ status: "ok", kind: "auth-refresh" });
    }
    expect(mocks.prepare).toHaveBeenCalledTimes(2);
    const [alpha, beta] = releases();
    expect(alpha).not.toHaveBeenCalled();
    expect(beta).not.toHaveBeenCalled();

    expect(await run(input("alpha", "replacement"))).toMatchObject({ status: "ok" });
    expect(alpha).toHaveBeenCalledTimes(1);
    expect(beta).not.toHaveBeenCalled();
    await run(input("beta"));
    await run(input("alpha", "replacement"));
    expect(mocks.prepare).toHaveBeenCalledTimes(3);

    await run(input("alpha"));
    expect(mocks.prepare).toHaveBeenCalledTimes(4);
    expect(releases()[2]).toHaveBeenCalledTimes(1);
    expect(alpha).toHaveBeenCalledTimes(1);
    expect(beta).not.toHaveBeenCalled();
  });

  it.each(["failed", "generation-mismatch"] as const)(
    "keeps a prior successful slot when its replacement is %s",
    async (status) => {
      const run = gatewayHandler();
      await run(input());
      const original = releases()[0]!;
      let changed = input("alpha", "replacement");
      if (status === "failed") {
        mocks.refreshAuth.mockImplementationOnce(() => {
          throw new Error("Synthetic auth refresh failure");
        });
      } else {
        changed = { ...changed, generationFingerprint: "different-owner-fingerprint" };
      }
      expect(await run(changed)).toMatchObject({ status });
      expect(original).not.toHaveBeenCalled();
      if (status === "failed") {
        expect(releases()[1]).toHaveBeenCalledTimes(1);
      } else {
        expect(mocks.discard).toHaveBeenCalledTimes(1);
        expect(releases()[1]).not.toHaveBeenCalled();
      }
      expect(await run(input())).toMatchObject({ status: "ok" });
      expect(mocks.prepare).toHaveBeenCalledTimes(2);
      expect(original).not.toHaveBeenCalled();
    },
  );

  it("detaches every old epoch slot before joining all releases, including a failed release", async () => {
    const run = gatewayHandler();
    await run(input("alpha"));
    await run(input("beta"));
    const [alpha, beta] = releases();
    alpha!.mockRejectedValue(new Error("Synthetic release failure"));
    const changed = input("alpha");
    changed.input.env = { ...changed.input.env, CATALOG_EPOCH: "next" };

    await expect(run(changed)).rejects.toThrow("Catalog agent generations failed to retire");
    expect(alpha).toHaveBeenCalledTimes(1);
    expect(beta).toHaveBeenCalledTimes(1);
    expect(mocks.prepare).toHaveBeenCalledTimes(2);

    expect(await run(changed)).toMatchObject({ status: "ok" });
    expect(mocks.prepare).toHaveBeenCalledTimes(3);
    expect(await run(input("beta"))).toMatchObject({ status: "ok" });
    expect(mocks.prepare).toHaveBeenCalledTimes(4);
    expect(alpha).toHaveBeenCalledTimes(1);
    expect(beta).toHaveBeenCalledTimes(1);
    expect(releases()[2]).toHaveBeenCalledTimes(1);
  });

  it.each([
    { label: "unconfigured agent", agentId: "transient", directory: agentDir("transient") },
    { label: "noncanonical directory", agentId: "alpha", directory: path.join(stateDir, "other") },
    { label: "unbound directory", agentId: undefined, directory: agentDir("alpha") },
  ])("keeps $label requests ephemeral", async ({ agentId, directory }) => {
    const run = gatewayHandler();
    // Explicitly omit the identity for the unbound case; input() defaults omitted ids to alpha.
    const value = input(agentId ?? "alpha", "initial", directory);
    if (!agentId) {
      delete value.input.agentId;
    }
    expect(await run(value)).toMatchObject({ status: "ok" });
    expect(await run(value)).toMatchObject({ status: "ok" });
    expect(mocks.prepare).toHaveBeenCalledTimes(2);
    for (const release of releases()) {
      expect(release).toHaveBeenCalledTimes(1);
    }
    await run(input("alpha"));
    await run(input("alpha"));
    expect(mocks.prepare).toHaveBeenCalledTimes(3);
    expect(releases()[2]).not.toHaveBeenCalled();
  });

  it("retains standalone work without consulting configured directory admission", async () => {
    const value = input("unconfigured", "initial", path.join(stateDir, "standalone"));
    const directory = vi.spyOn(agentScope, "resolveEffectiveAgentDir").mockImplementation(() => {
      throw new Error("Standalone must not resolve configured directories");
    });
    const data: PreparedModelCatalogWorkerData = {
      ...value,
      sourceCaptureDirectory: path.join(stateDir, "captures"),
    };
    const run = createPreparedCatalogTaskHandler(data);
    expect(await run(request)).toMatchObject({ status: "ok" });
    expect(await run(request)).toMatchObject({ status: "ok" });
    expect(mocks.prepare).toHaveBeenCalledTimes(1);
    expect(releases()[0]).not.toHaveBeenCalled();
    expect(directory).not.toHaveBeenCalled();
  });
});
