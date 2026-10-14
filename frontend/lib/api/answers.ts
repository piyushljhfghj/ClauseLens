/**
 * Grounded question answering over the selected documents (Phase 5).
 *
 * Read `grounding` before showing `answer`:
 * - "grounded": every cited statement has a verified quotation;
 * - "partially_grounded": some citations could not be verified (see `notes`);
 * - "ungrounded": nothing could be verified, so `answer` states that support was insufficient;
 * - "not_found": the selected documents did not contain relevant evidence.
 * Only citations with status "verified" carry a quote and a location; their text is the stored
 * canonical document text, never the model's wording. No UI calls this yet (chat is Phase 8).
 */

import { apiRequest, type RequestOptions } from "@/lib/api/client";
import type { components } from "@/lib/api/schema";

export type AnswerRequest = components["schemas"]["AnswerRequest"];
export type AnswerResponse = components["schemas"]["AnswerResponse"];
export type Citation = components["schemas"]["CitationOut"];
export type Evidence = components["schemas"]["EvidenceOut"];

// The LLM call can take a while; allow more than the default request timeout.
const ANSWER_TIMEOUT_MS = 90_000;

export function askDocuments(
  request: AnswerRequest,
  options: Pick<RequestOptions, "signal" | "baseUrl" | "fetchImpl" | "timeoutMs"> = {},
): Promise<AnswerResponse> {
  return apiRequest<AnswerResponse>("/api/answers", {
    timeoutMs: ANSWER_TIMEOUT_MS,
    ...options,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
}

/** Citations that may be shown as genuine quotations. */
export function verifiedCitations(response: Pick<AnswerResponse, "citations">): Citation[] {
  return response.citations.filter((c) => c.status === "verified");
}
