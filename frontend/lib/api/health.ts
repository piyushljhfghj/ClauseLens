import { apiRequest, type RequestOptions } from "@/lib/api/client";
import type { components } from "@/lib/api/schema";

export type HealthResponse = components["schemas"]["HealthResponse"];

/** Readiness check. A 503 still carries a HealthResponse describing which check failed. */
export function getHealth(options: RequestOptions = {}): Promise<HealthResponse> {
  return apiRequest<HealthResponse>("/api/health", { ...options, acceptStatuses: [503] });
}
