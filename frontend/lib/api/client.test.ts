import { describe, expect, it } from "vitest";

import { ApiError, apiRequest } from "./client";

function fakeFetch(status: number, body: unknown, headers: Record<string, string> = {}): typeof fetch {
  return async () =>
    new Response(body === undefined ? null : JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json", ...headers },
    });
}

const base = { baseUrl: "http://api.test" };

describe("apiRequest", () => {
  it("returns the parsed body on success", async () => {
    const result = await apiRequest<{ ok: boolean }>("/x", { ...base, fetchImpl: fakeFetch(200, { ok: true }) });
    expect(result).toEqual({ ok: true });
  });

  it("turns the backend error envelope into an ApiError", async () => {
    const envelope = { error: { code: "DOCUMENT_NOT_FOUND", message: "Gone.", details: { id: "1" }, request_id: "r1" } };
    const error = await apiRequest("/x", { ...base, fetchImpl: fakeFetch(404, envelope) }).catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ code: "DOCUMENT_NOT_FOUND", message: "Gone.", status: 404, requestId: "r1" });
  });

  it("handles non-envelope error bodies", async () => {
    const error = await apiRequest("/x", { ...base, fetchImpl: fakeFetch(502, { nope: 1 }) }).catch((e) => e);
    expect(error).toMatchObject({ code: "HTTP_ERROR", status: 502 });
  });

  it("returns the body for explicitly accepted statuses", async () => {
    const result = await apiRequest<{ status: string }>("/x", {
      ...base,
      acceptStatuses: [503],
      fetchImpl: fakeFetch(503, { status: "degraded" }),
    });
    expect(result.status).toBe("degraded");
  });

  it("reports an unreachable API as NETWORK_ERROR", async () => {
    const failing: typeof fetch = async () => {
      throw new TypeError("Failed to fetch");
    };
    const error = await apiRequest("/x", { ...base, fetchImpl: failing }).catch((e) => e);
    expect(error).toMatchObject({ code: "NETWORK_ERROR" });
  });

  it("reports a slow API as TIMEOUT", async () => {
    const hanging: typeof fetch = (_input, init) =>
      new Promise((_, reject) => init?.signal?.addEventListener("abort", () => reject(init.signal?.reason)));
    const error = await apiRequest("/x", { ...base, timeoutMs: 20, fetchImpl: hanging }).catch((e) => e);
    expect(error).toMatchObject({ code: "TIMEOUT" });
  });
});
