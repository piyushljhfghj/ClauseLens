/**
 * Minimal typed HTTP client for the ClauseLens API.
 *
 * Every failure becomes an `ApiError` with a stable `code`, whether it came from the backend's
 * error envelope, a non-JSON response, a timeout, or the network (API down, CORS, DNS).
 */

import { API_BASE_URL } from "@/lib/config";

const DEFAULT_TIMEOUT_MS = 15_000;

export class ApiError extends Error {
  constructor(
    public readonly code: string,
    message: string,
    public readonly status: number | null = null,
    public readonly requestId: string | null = null,
    public readonly details: Record<string, unknown> = {},
  ) {
    super(message);
    this.name = "ApiError";
  }
}

interface ErrorEnvelope {
  error: {
    code: string;
    message: string;
    details?: Record<string, unknown>;
    request_id?: string;
  };
}

function isErrorEnvelope(body: unknown): body is ErrorEnvelope {
  if (typeof body !== "object" || body === null || !("error" in body)) return false;
  const error = (body as { error: unknown }).error;
  return (
    typeof error === "object" &&
    error !== null &&
    typeof (error as { code?: unknown }).code === "string" &&
    typeof (error as { message?: unknown }).message === "string"
  );
}

export interface RequestOptions extends Omit<RequestInit, "signal"> {
  timeoutMs?: number;
  signal?: AbortSignal;
  /** Extra non-2xx statuses whose JSON body should be returned instead of thrown. */
  acceptStatuses?: number[];
  baseUrl?: string;
  fetchImpl?: typeof fetch;
}

export async function apiRequest<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const {
    timeoutMs = DEFAULT_TIMEOUT_MS,
    signal,
    acceptStatuses = [],
    baseUrl = API_BASE_URL,
    fetchImpl = fetch,
    ...init
  } = options;

  const timeoutSignal = AbortSignal.timeout(timeoutMs);
  const combinedSignal = signal ? AbortSignal.any([signal, timeoutSignal]) : timeoutSignal;

  let response: Response;
  try {
    response = await fetchImpl(`${baseUrl}${path}`, {
      ...init,
      headers: { Accept: "application/json", ...init.headers },
      signal: combinedSignal,
    });
  } catch (err) {
    if (signal?.aborted) throw err; // caller cancelled on purpose; let them handle it
    if (timeoutSignal.aborted) {
      throw new ApiError("TIMEOUT", `The server did not respond within ${timeoutMs / 1000}s.`);
    }
    throw new ApiError(
      "NETWORK_ERROR",
      `Could not reach the ClauseLens API at ${baseUrl}. Is the backend running?`,
    );
  }

  const requestId = response.headers.get("X-Request-ID");
  const body: unknown = await response.json().catch(() => null);

  if (response.ok || acceptStatuses.includes(response.status)) {
    if (body === null && response.status !== 204) {
      throw new ApiError("INVALID_RESPONSE", "The server returned an unreadable response.", response.status, requestId);
    }
    return body as T;
  }

  throw toApiError(response.status, body, requestId);
}

/** Convert a non-2xx response into an ApiError (shared by fetch and XHR code paths). */
export function toApiError(status: number, body: unknown, requestId: string | null): ApiError {
  if (isErrorEnvelope(body)) {
    return new ApiError(
      body.error.code,
      body.error.message,
      status,
      body.error.request_id ?? requestId,
      body.error.details ?? {},
    );
  }
  return new ApiError("HTTP_ERROR", `Request failed with status ${status}.`, status, requestId);
}
