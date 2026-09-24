import fs from "node:fs";
import path from "node:path";
import { threadId } from "node:worker_threads";
import { expect, it, vi } from "vitest";
import { saveAuthProfileStore } from "./auth-profiles/store-runtime.js";
import type { AuthProfileStore } from "./auth-profiles/types.js";
import { getPreparedModelCatalogWorkerPoolSnapshot } from "./prepared-model-catalog-worker.js";
import {
  EXTERNAL_AUTH_PROFILE_ID,
  PROVIDER_ID,
} from "./prepared-model-catalog-worker.test-support.js";
import {
  getPreparedModelFullCatalogAuth,
  loadPreparedModelRuntimeAuth,
} from "./prepared-model-runtime-auth.js";
import { closePreparedModelRuntimeSnapshots } from "./prepared-model-runtime.lifecycle.js";
import { createCatalogFleetFixture } from "./test-helpers/prepared-model-catalog-fleet-fixture.js";
import {
  loadCompletedFullCatalog,
  readCatalogDiscoveryCaptures,
  usePreparedCatalogWorkerFixtures,
} from "./test-helpers/prepared-model-catalog-worker-fixture.js";

const { makeTempDir } = usePreparedCatalogWorkerFixtures();
const createFleetFixture = createCatalogFleetFixture(makeTempDir);

it("reuses each agent's four discovery scopes while reading fresh credentials", async () => {
  vi.stubEnv("CODEX_HOME", makeTempDir("openclaw-reuse-empty-codex-"));
  const agentIds = ["fleet-a", "fleet-b"];
  const providerIds = Array.from({ length: 4 }, (_, index) => `worker-reuse-${index}`);
  const profileId = (provider: string) => `${provider}:reuse`;
  const credential = (agentId: string, revision: string) => `synthetic-${agentId}-${revision}`;
  const writeAuth = (stateDir: string, revision: string) => {
    for (const agentId of agentIds) {
      const profiles: AuthProfileStore["profiles"] = {
        [`fleet:${agentId}`]: {
          type: "api_key",
          provider: "fleet-proof",
          key: `synthetic-${agentId}`,
        },
      };
      for (const provider of providerIds) {
        profiles[profileId(provider)] = {
          type: "api_key",
          provider,
          key: credential(agentId, revision),
        };
      }
      saveAuthProfileStore(
        { version: 1, profiles },
        path.join(stateDir, "agents", agentId, "agent"),
      );
    }
  };
  const fixture = await createFleetFixture(
    (seed) => {
      const registrations = path.join(seed.root, "worker-registrations.jsonl");
      const executions = path.join(seed.root, "worker-catalog-executions.jsonl");
      const bundledRoot = path.join(seed.root, "bundled");
      const fixtureEnv: NodeJS.ProcessEnv = seed.env;
      fs.mkdirSync(bundledRoot);
      // Nonbundled provider plugins are always eligible for runtime augmentation,
      // which would put every fixture in the base registry and hide scope churn.
      for (const [name, value] of Object.entries({
        OPENCLAW_DISABLE_BUNDLED_PLUGINS: "0",
        OPENCLAW_BUNDLED_PLUGINS_DIR: bundledRoot,
        OPENCLAW_TEST_TRUST_BUNDLED_PLUGINS_DIR: "1",
      })) {
        fixtureEnv[name] = value;
        vi.stubEnv(name, value);
      }
      const recordRegistration = (pluginId: string) => `
    if (require("node:worker_threads").threadId !== ${threadId}) {
      fs.appendFileSync(${JSON.stringify(registrations)}, JSON.stringify({
        pluginId: ${JSON.stringify(pluginId)}, filename: __filename,
      }) + "\\n");
    }`;
      // Module paths can remain cached even when the worker reconstructs their registries.
      // Record registration, rather than module evaluation, to expose both eviction loops.
      const baseEntry = path.join(seed.root, "plugin", "index.cjs");
      fs.writeFileSync(
        baseEntry,
        fs
          .readFileSync(baseEntry, "utf8")
          .replace("  register(api) {", `  register(api) {${recordRegistration(PROVIDER_ID)}`),
      );
      for (const provider of providerIds) {
        const directory = path.join(bundledRoot, provider);
        fs.mkdirSync(directory);
        const entry = path.join(directory, "index.cjs");
        fs.writeFileSync(
          entry,
          `const fs = require("node:fs");
module.exports = { id: ${JSON.stringify(provider)}, register(api) {
  ${recordRegistration(provider)}
  api.registerProvider({
    id: ${JSON.stringify(provider)}, label: "Scoped reuse fixture", auth: [],
    catalog: { run(context) {
      fs.appendFileSync(${JSON.stringify(executions)}, JSON.stringify({
        provider: ${JSON.stringify(provider)}, agentDir: context.agentDir,
        threadId: require("node:worker_threads").threadId,
      }) + "\\n");
      const auth = context.resolveProviderApiKey(${JSON.stringify(provider)});
      return { provider: {
        api: "openai-completions", baseUrl: "https://reuse.invalid/v1",
        models: [{ id: "auth-" + auth.discoveryApiKey, name: "Fresh credential model" }],
      } };
    } },
  });
} };
`,
        );
        fs.writeFileSync(
          path.join(directory, "openclaw.plugin.json"),
          JSON.stringify({
            id: provider,
            providers: [provider],
            modelCatalog: { discovery: { [provider]: "runtime" } },
            configSchema: { type: "object", additionalProperties: false },
          }),
        );
        fs.writeFileSync(
          path.join(directory, "package.json"),
          JSON.stringify({
            name: `@openclaw/${provider}`,
            version: "0.0.0",
            openclaw: { extensions: ["./index.cjs"] },
          }),
        );
        seed.config.plugins.allow.push(provider);
        Object.assign(seed.config.plugins.entries, { [provider]: { enabled: true } });
      }
      writeAuth(seed.env.OPENCLAW_STATE_DIR!, "A");
    },
    false,
    { agentCount: agentIds.length },
  );
  for (const snapshot of fixture.snapshots) {
    for (const provider of providerIds) {
      expect(snapshot.metadataSnapshot.plugins.find(({ id }) => id === provider)).toMatchObject({
        origin: "bundled",
        modelCatalog: { discovery: { [provider]: "runtime" } },
      });
    }
    await loadCompletedFullCatalog(snapshot);
  }
  const workerExecutions = () =>
    fs
      .readFileSync(path.join(fixture.root, "worker-catalog-executions.jsonl"), "utf8")
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line) as { provider: string; agentDir: string; threadId: number })
      .filter((execution) => execution.threadId !== threadId);
  const refreshScopes = async (revision: string) => {
    for (const provider of providerIds) {
      for (const [index, snapshot] of fixture.snapshots.entries()) {
        const completedExecutions = workerExecutions().length;
        await snapshot.loadFullModelCatalog!({ providerIds: [provider], refresh: true });
        const catalog = await loadCompletedFullCatalog(snapshot);
        expect(workerExecutions().slice(completedExecutions)).toEqual([
          { provider, agentDir: snapshot.agentDir, threadId: expect.any(Number) },
        ]);
        const key = credential(agentIds[index]!, revision);
        expect(
          catalog.entries.filter((entry) => entry.provider === provider).map(({ id }) => id),
        ).toEqual([`auth-${key}`]);
        expect(
          getPreparedModelFullCatalogAuth(catalog)?.authStore.profiles[profileId(provider)],
        ).toMatchObject({ key });
      }
    }
  };
  await refreshScopes("A");
  const readRegistrations = () =>
    fs.readFileSync(path.join(fixture.root, "worker-registrations.jsonl"), "utf8");
  const registrations = readRegistrations();
  const registeredPluginIds = new Set(
    registrations
      .trim()
      .split("\n")
      .map((line) => (JSON.parse(line) as { pluginId: string }).pluginId),
  );
  expect(registeredPluginIds).toEqual(new Set([PROVIDER_ID, ...providerIds]));
  const workerCaptures = () =>
    new Set(
      readCatalogDiscoveryCaptures(fixture.root)
        .filter((capture) => capture.threadId !== threadId)
        .map((capture) => capture.filename),
    );
  const captures = workerCaptures();
  expect(captures.size).toBeGreaterThan(0);
  // Durable auth publication replaces the snapshot itself. External auth refresh
  // is request-local, so it exercises fresh credentials on these retained owners.
  fs.writeFileSync(fixture.externalAuthPath, "B");
  for (const snapshot of fixture.snapshots) {
    const auth = await loadPreparedModelRuntimeAuth(snapshot, { providerIds: [PROVIDER_ID] });
    expect(auth?.authStore.profiles[EXTERNAL_AUTH_PROFILE_ID]).toMatchObject({ access: "v1:B" });
  }
  await refreshScopes("A");
  expect(readRegistrations()).toBe(registrations);
  expect(workerCaptures()).toEqual(captures);
  expect(getPreparedModelCatalogWorkerPoolSnapshot()).toMatchObject({
    workers: 1,
    workersCreated: 1,
    activeTasks: 0,
    pendingTasks: 0,
  });
  const filename = [...captures][0]!;
  const captureRoot = filename.slice(0, filename.indexOf(`${path.sep}openclaw-plugin-build-`));
  expect(path.basename(captureRoot)).toMatch(/^openclaw-model-catalog-/);
  await closePreparedModelRuntimeSnapshots();
  expect(fs.existsSync(captureRoot)).toBe(false);
});
