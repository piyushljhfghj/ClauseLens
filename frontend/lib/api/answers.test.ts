import { describe, expect, it } from "vitest";

import { askDocuments, verifiedCitations, type AnswerResponse, type Citation } from "./answers";

const citation = (n: number, status: "verified" | "unverified"): Citation => ({
  n,
  status,
  proposed_quote: "q",
  cited_evidence_id: "E1",
  location_ambiguous: false,
});

describe("askDocuments", () => {
  it("POSTs the question and selected documents", async () => {
    let body: unknown = null;
    let url = "";
    const fetchImpl: typeof fetch = async (input, init) => {
      url = String(input);
      body = JSON.parse(String(init?.body));
      const response: Partial<AnswerResponse> = { grounding: "not_found", citations: [] };
      return new Response(JSON.stringify(response), { status: 200 });
    };
    const result = await askDocuments(
      { question: "What is the cap?", document_ids: ["d1"] },
      { baseUrl: "http://api.test", fetchImpl },
    );
    expect(url).toBe("http://api.test/api/answers");
    expect(body).toEqual({ question: "What is the cap?", document_ids: ["d1"] });
    expect(result.grounding).toBe("not_found");
  });

  it("surfaces a missing LLM configuration as a typed error", async () => {
    const fetchImpl: typeof fetch = async () =>
      new Response(JSON.stringify({ error: { code: "LLM_NOT_CONFIGURED", message: "Not configured." } }), {
        status: 503,
      });
    const error = await askDocuments({ question: "q", document_ids: ["d"] }, { baseUrl: "http://x", fetchImpl }).catch(
      (e) => e,
    );
    expect(error).toMatchObject({ code: "LLM_NOT_CONFIGURED", status: 503 });
  });
});

describe("verifiedCitations", () => {
  it("keeps only verified citations", () => {
    const response = { citations: [citation(1, "verified"), citation(2, "unverified")] };
    expect(verifiedCitations(response).map((c) => c.n)).toEqual([1]);
  });
});
