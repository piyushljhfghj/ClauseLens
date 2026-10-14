"use client";

import { useCallback, useEffect, useState } from "react";

import { ApiError } from "@/lib/api/client";
import { getHealth, type HealthResponse } from "@/lib/api/health";
import { API_BASE_URL } from "@/lib/config";

const POLL_INTERVAL_MS = 15_000;

type State =
  | { kind: "loading" }
  | { kind: "ok"; health: HealthResponse; checkedAt: Date }
  | { kind: "unreachable"; error: ApiError; checkedAt: Date };

function Dot({ tone }: { tone: "ok" | "warn" | "error" | "idle" }) {
  const colour = {
    ok: "bg-emerald-500",
    warn: "bg-amber-500",
    error: "bg-red-500",
    idle: "bg-zinc-300 dark:bg-zinc-600 animate-pulse",
  }[tone];
  return <span aria-hidden className={`inline-block h-2.5 w-2.5 shrink-0 rounded-full ${colour}`} />;
}

function Row({ label, ok, detail }: { label: string; ok: boolean | null; detail?: string | null }) {
  return (
    <li className="flex items-start gap-3 py-2.5">
      <span className="mt-1.5">
        <Dot tone={ok === null ? "idle" : ok ? "ok" : "error"} />
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline justify-between gap-4">
          <span className="text-sm font-medium">{label}</span>
          <span className="text-xs text-zinc-500">{ok === null ? "checking…" : ok ? "OK" : "Failing"}</span>
        </div>
        {detail ? <p className="mt-0.5 break-words font-mono text-xs text-zinc-500">{detail}</p> : null}
      </div>
    </li>
  );
}

export function BackendStatus() {
  const [state, setState] = useState<State>({ kind: "loading" });

  const check = useCallback(async (signal?: AbortSignal) => {
    try {
      const health = await getHealth({ signal });
      setState({ kind: "ok", health, checkedAt: new Date() });
    } catch (err) {
      if (signal?.aborted) return;
      const error = err instanceof ApiError ? err : new ApiError("UNKNOWN", String(err));
      setState({ kind: "unreachable", error, checkedAt: new Date() });
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    // Deferred so the first state update happens in a callback, not synchronously in the effect.
    const first = setTimeout(() => void check(controller.signal), 0);
    const timer = setInterval(() => void check(controller.signal), POLL_INTERVAL_MS);
    return () => {
      controller.abort();
      clearTimeout(first);
      clearInterval(timer);
    };
  }, [check]);

  const health = state.kind === "ok" ? state.health : null;
  const apiReachable = state.kind === "loading" ? null : state.kind === "ok";
  const overall =
    state.kind === "loading" ? "idle" : state.kind === "unreachable" ? "error" : health?.status === "ok" ? "ok" : "warn";
  const headline = {
    idle: "Checking backend…",
    ok: "All systems operational",
    warn: "Backend reachable, but degraded",
    error: "Backend unreachable",
  }[overall];

  return (
    <section
      aria-labelledby="status-heading"
      className="rounded-xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900"
    >
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <Dot tone={overall} />
          <h2 id="status-heading" className="text-base font-semibold">
            {headline}
          </h2>
        </div>
        <button
          type="button"
          onClick={() => void check()}
          className="rounded-md border border-zinc-200 px-3 py-1.5 text-xs font-medium hover:bg-zinc-50 dark:border-zinc-700 dark:hover:bg-zinc-800"
        >
          Re-check
        </button>
      </div>

      <p className="mt-1 text-xs text-zinc-500">
        API: <span className="font-mono">{API_BASE_URL}</span>
        {state.kind !== "loading" ? <> · checked {state.checkedAt.toLocaleTimeString()}</> : null}
        {health ? <> · v{health.version} ({health.environment})</> : null}
      </p>

      <ul className="mt-4 divide-y divide-zinc-100 dark:divide-zinc-800">
        <Row
          label="API server"
          ok={apiReachable}
          detail={state.kind === "unreachable" ? `${state.error.code}: ${state.error.message}` : null}
        />
        <Row
          label="PostgreSQL"
          ok={health ? health.checks.database.status === "ok" : apiReachable === false ? false : null}
          detail={
            health
              ? health.checks.database.status === "ok"
                ? `${health.checks.database.latency_ms} ms`
                : health.checks.database.detail
              : null
          }
        />
        <Row
          label="pgvector extension"
          ok={health ? health.checks.pgvector.status === "ok" : apiReachable === false ? false : null}
          detail={health?.checks.pgvector.detail}
        />
        <Row
          label="Database migrations"
          ok={health ? health.checks.migrations.status === "ok" : apiReachable === false ? false : null}
          detail={health?.checks.migrations.detail}
        />
        <Row
          label="OCR (Tesseract)"
          ok={health ? health.checks.ocr.status !== "error" : apiReachable === false ? false : null}
          detail={health?.checks.ocr.detail}
        />
        <Row
          label="Embedding model"
          ok={health ? health.checks.embeddings.status === "ok" : apiReachable === false ? false : null}
          detail={health?.checks.embeddings.detail}
        />
      </ul>

      {health && !health.llm_configured ? (
        <p className="mt-4 rounded-md bg-zinc-50 px-3 py-2 text-xs text-zinc-600 dark:bg-zinc-800/60 dark:text-zinc-400">
          LLM provider not configured yet (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL). Not needed until chat is built.
        </p>
      ) : null}
    </section>
  );
}
