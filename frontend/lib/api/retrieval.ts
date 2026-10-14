/**
 * Hybrid retrieval over the selected documents (Phase 4).
 *
 * Returns ranked passages with their exact canonical source location and extraction quality.
 * Retrieval says a passage is *likely relevant*; it does not prove that it supports an answer.
 * No UI calls this yet: question answering (Phase 5) and chat (Phase 8) will.
 */

import { apiRequest, type RequestOptions } from "@/lib/api/client";
import type { components } from "@/lib/api/schema";

export type RetrievalRequest = components["schemas"]["RetrievalRequest"];
export type RetrievalResponse = components["schemas"]["RetrievalResponse"];
export type RetrievedPassage = components["schemas"]["RetrievedPassage"];
export type PassageQuality = components["schemas"]["PassageQuality"];

export function searchDocuments(
  request: RetrievalRequest,
  options: Pick<RequestOptions, "signal" | "baseUrl" | "fetchImpl"> = {},
): Promise<RetrievalResponse> {
  return apiRequest<RetrievalResponse>("/api/retrieval/search", {
    ...options,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
}
