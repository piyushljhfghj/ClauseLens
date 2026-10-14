import { describe, expect, it } from "vitest";

import { ApiError } from "./client";
import { searchDocuments, type RetrievalResponse } from "./retrieval";

const empty: RetrievalResponse = {
  query: "liability cap",
  passages: [],
  no_match: true,
  coverage: [],
  analysis: { terms: ["liability", "cap"], phrases: [], section_refs: [] },
  notes: [],
};

describe("searchDocuments", () => {
  it("POSTs the query and selected documents as JSON", async () => {
    let seen: { url: string; init: RequestInit } | null = null;
    const fetchImpl: typeof fetch = async (input, init) => {
      seen = { url: String(input), init: init ?? {} };
      return new Response(JSON.stringify(empty), { status: 200, headers: { "Content-Type": "application/json" } });
    };
    const result = await searchDocuments(
      { query: "liability cap", document_ids: ["a1b2c3d4-0000-0000-0000-000000000001"], top_k: 5 },
      { baseUrl: "http://api.test", fetchImpl },
    );
    expect(result.no_match).toBe(true);
    expect(seen!.url).toBe("http://api.test/api/retrieval/search");
    expect(seen!.init.method).toBe("POST");
    expect(JSON.parse(String(seen!.init.body))).toEqual({
      query: "liability cap",
      document_ids: ["a1b2c3d4-0000-0000-0000-000000000001"],
      top_k: 5,
    });
  });

  it("surfaces the not-indexed error code", async () => {
    const fetchImpl: typeof fetch = async () =>
      new Response(
        JSON.stringify({ error: { code: "DOCUMENT_NOT_INDEXED", message: "Reprocess them.", details: {} } }),
        { status: 409 },
      );
    const error = await searchDocuments({ query: "x", document_ids: ["d"] }, { baseUrl: "http://api.test", fetchImpl }).catch(
      (e) => e,
    );
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ code: "DOCUMENT_NOT_INDEXED", status: 409 });
  });
});
