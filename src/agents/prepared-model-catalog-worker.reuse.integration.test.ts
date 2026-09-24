import fs from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { threadId } from "node:worker_threads";
import { expect, it, vi } from "vitest";
import { saveAuthProfileStore } from "./auth-profiles/store-runtime.js";
import type { AuthProfileStore } from "./auth-profiles/types.js";
import { MAX_CATALOG_WORKER_REGISTRIES } from "./prepared-model-catalog-worker.generations.js";
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

it(
  "reuses exact discovery scopes across agents while reading fresh credentials",
  async () => {
    const fleetProof = process.env.OPENCLAW_CATALOG_FLEET_PROOF === "1";
    const started = performance.now();
    vi.stubEnv("CODEX_HOME", makeTempDir("openclaw-reuse-empty-codex-"));
    const agentIds = Array.from(
      { length: fleetProof ? 28 : 2 },
      (_, index) => `fleet-${index < 26 ? String.fromCharCode(97 + index) : index + 1}`,
    );
    const providerIds = Array.from({ length: 6 }, (_, index) => `worker-reuse-${index}`);
    const profileId = (provider: string) => `${provider}:reuse`;
    const credential = (agentId: string, revision: string) => `synthetic-${agentId}-${revision}`;
    const checkRetainedPayload = fleetProof
      ? `if (retainedPayload && retainedPayload[0] !== 7) throw new Error("Lost retained fixture payload");`
      : "";
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
        const weightedMarker = path.join(seed.root, "weighted-registrations");
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
    ${fleetProof ? `const retainedPayload = require("node:worker_threads").threadId !== ${threadId} && fs.existsSync(${JSON.stringify(weightedMarker)}) ? new Array(1024 * 1024).fill(7) : undefined;` : ""}
    if (require("node:worker_threads").threadId !== ${threadId}) {
      fs.appendFileSync(${JSON.stringify(registrations)}, JSON.stringify({
        pluginId: ${JSON.stringify(pluginId)}, filename: __filename,
        ${fleetProof ? "weighted: retainedPayload !== undefined," : ""}
      }) + "\\n");
    }`;
        // Module paths can remain cached even when the worker reconstructs their registries.
        // Record registration, rather than module evaluation, to expose both eviction loops.
        const baseEntry = path.join(seed.root, "plugin", "index.cjs");
        fs.writeFileSync(
          baseEntry,
          fs
            .readFileSync(baseEntry, "utf8")
            .replace("  register(api) {", `  register(api) {${recordRegistration(PROVIDER_ID)}`)
            .replace("run(context) {", `run(context) {${checkRetainedPayload}`),
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
      ${checkRetainedPayload}
      fs.appendFileSync(${JSON.stringify(executions)}, JSON.stringify({
        provider: ${JSON.stringify(provider)}, agentDir: context.agentDir,
        threadId: require("node:worker_threads").threadId,
        ${fleetProof ? 'heap: require("node:v8").getHeapStatistics(), resourceLimits: require("node:worker_threads").resourceLimits, nodeOptions: process.env.NODE_OPTIONS ?? null,' : ""}
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
    const workerCaptures = () =>
      new Set(
        readCatalogDiscoveryCaptures(fixture.root)
          .filter((capture) => capture.threadId !== threadId)
          .map((capture) => capture.filename),
      );
    const captureRoot = () => {
      const filename = [...workerCaptures()][0]!;
      return filename.slice(0, filename.indexOf(`${path.sep}openclaw-plugin-build-`));
    };
    const captureDirectories = () =>
      fs.readdirSync(captureRoot()).filter((name) => name.startsWith("openclaw-plugin-build-"));
    const bytesUnder = (directory: string): number =>
      fs.readdirSync(directory, { withFileTypes: true }).reduce((bytes, entry) => {
        const filename = path.join(directory, entry.name);
        return bytes + (entry.isDirectory() ? bytesUnder(filename) : fs.lstatSync(filename).size);
      }, 0);
    const reportFleet = (stage: string) => {
      if (!fleetProof) {
        return;
      }
      const root = captureRoot();
      const executionsPath = path.join(fixture.root, "worker-catalog-executions.jsonl");
      const executions: {
        threadId: number;
        heap: { used_heap_size: number; heap_size_limit: number };
        resourceLimits: { maxOldGenerationSizeMb: number };
        nodeOptions: string | null;
      }[] = fs.existsSync(executionsPath)
        ? fs
            .readFileSync(executionsPath, "utf8")
            .trim()
            .split("\n")
            .map((line) => JSON.parse(line))
            .filter((execution: { threadId: number }) => execution.threadId !== threadId)
        : [];
      const registrations = fs
        .readFileSync(path.join(fixture.root, "worker-registrations.jsonl"), "utf8")
        .trim()
        .split("\n")
        .map((line) => JSON.parse(line) as { weighted?: boolean });
      expect(executions.at(-1)?.resourceLimits.maxOldGenerationSizeMb).toBe(512);
      expect(executions.at(-1)?.nodeOptions).toBeNull();
      console.log(
        JSON.stringify({
          stage,
          agents: agentIds.length,
          scopes: providerIds.length,
          registrations: registrations.length,
          weightedRegistrations: registrations.filter(({ weighted }) => weighted).length,
          capturesRetained: captureDirectories().length,
          captureBytes: bytesUnder(root),
          workerHeap: executions.at(-1)?.heap,
          resourceLimits: executions.at(-1)?.resourceLimits,
          maxSampledWorkerHeap: Math.max(
            0,
            ...executions.map(({ heap }) => heap?.used_heap_size ?? 0),
          ),
          rss: process.memoryUsage().rss,
          elapsedMs: performance.now() - started,
          pool: getPreparedModelCatalogWorkerPoolSnapshot(),
        }),
      );
    };
    for (const [index, snapshot] of fixture.snapshots.entries()) {
      for (const provider of providerIds) {
        expect(snapshot.metadataSnapshot.plugins.find(({ id }) => id === provider)).toMatchObject({
          origin: "bundled",
          modelCatalog: { discovery: { [provider]: "runtime" } },
        });
      }
      await loadCompletedFullCatalog(snapshot);
      if ([0, 3, 12, 27].includes(index)) {
        reportFleet(`base-${index + 1}`);
      }
    }
    const workerExecutions = () =>
      fs
        .readFileSync(path.join(fixture.root, "worker-catalog-executions.jsonl"), "utf8")
        .trim()
        .split("\n")
        .map((line) => JSON.parse(line) as { provider: string; agentDir: string; threadId: number })
        .filter((execution) => execution.threadId !== threadId)
        .map(({ provider, agentDir, threadId }) => ({ provider, agentDir, threadId }));
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
          if (fleetProof) {
            // This fixture shares one base source capture; each retained discovery
            // owns one additional capture. Real plugins may have different sizes.
            expect(captureDirectories().length).toBeLessThanOrEqual(
              MAX_CATALOG_WORKER_REGISTRIES + 1,
            );
          }
        }
      }
    };
    await refreshScopes("A");
    reportFleet("scopes-first-pass");
    const firstPassCaptureDirectories = captureDirectories();
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
    reportFleet("scopes-second-pass");
    if (!fleetProof) {
      expect(readRegistrations()).toBe(registrations);
      expect(workerCaptures()).toEqual(captures);
    } else {
      const retained = new Set(captureDirectories());
      const retired = firstPassCaptureDirectories.filter((directory) => !retained.has(directory));
      expect(retired.length).toBeGreaterThan(0);
      console.log(
        JSON.stringify({
          stage: "physical-retirement",
          priorCaptures: firstPassCaptureDirectories.length,
          retiredCaptures: retired.length,
          retainedCaptures: retained.size,
        }),
      );
      // Opt-in capacity proof: payloads are synthetic JS arrays retained by each
      // worker registration's catalog closure, not representative plugin memory.
      fs.writeFileSync(path.join(fixture.root, "weighted-registrations"), "8 MiB per registration");
      await refreshScopes("A");
      reportFleet("weighted-first-pass");
      await refreshScopes("A");
      reportFleet("weighted-second-pass");
    }
    expect(getPreparedModelCatalogWorkerPoolSnapshot()).toMatchObject({
      workers: 1,
      workersCreated: 1,
      activeTasks: 0,
      pendingTasks: 0,
    });
    const root = captureRoot();
    expect(path.basename(root)).toMatch(/^openclaw-model-catalog-/);
    await closePreparedModelRuntimeSnapshots();
    expect(fs.existsSync(root)).toBe(false);
  },
  process.env.OPENCLAW_CATALOG_FLEET_PROOF === "1" ? 300_000 : undefined,
);
