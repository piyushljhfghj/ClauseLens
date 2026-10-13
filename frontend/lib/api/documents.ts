import { ApiError, apiRequest, toApiError } from "@/lib/api/client";
import type { components } from "@/lib/api/schema";
import { API_BASE_URL } from "@/lib/config";

export type DocumentOut = components["schemas"]["DocumentOut"];
export type DocumentList = components["schemas"]["DocumentList"];
export type DocumentStatus = DocumentOut["status"];

export function listDocuments(signal?: AbortSignal): Promise<DocumentList> {
  return apiRequest<DocumentList>("/api/documents", { signal });
}

export function getDocument(id: string, signal?: AbortSignal): Promise<DocumentOut> {
  return apiRequest<DocumentOut>(`/api/documents/${encodeURIComponent(id)}`, { signal });
}

export async function deleteDocument(id: string): Promise<void> {
  await apiRequest<null>(`/api/documents/${encodeURIComponent(id)}`, { method: "DELETE" });
}

export function reprocessDocument(id: string): Promise<DocumentOut> {
  return apiRequest<DocumentOut>(`/api/documents/${encodeURIComponent(id)}/reprocess`, {
    method: "POST",
  });
}

export function documentFileUrl(id: string, { download = false } = {}): string {
  const url = `${API_BASE_URL}/api/documents/${encodeURIComponent(id)}/file`;
  return download ? `${url}?download=true` : url;
}

export interface UploadHandle {
  promise: Promise<DocumentOut>;
  abort: () => void;
}

/**
 * Upload with byte-level progress. `fetch` cannot report upload progress, so this uses
 * XMLHttpRequest; errors are normalised to the same ApiError the rest of the client uses.
 */
export function uploadDocument(
  file: File,
  onProgress: (fraction: number) => void,
  { baseUrl = API_BASE_URL }: { baseUrl?: string } = {},
): UploadHandle {
  const xhr = new XMLHttpRequest();
  const promise = new Promise<DocumentOut>((resolve, reject) => {
    xhr.open("POST", `${baseUrl}/api/documents`);
    xhr.responseType = "text";
    xhr.setRequestHeader("Accept", "application/json");

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    };
    xhr.onload = () => {
      let body: unknown = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        // fall through with body = null
      }
      const requestId = xhr.getResponseHeader("X-Request-ID");
      if (xhr.status >= 200 && xhr.status < 300 && body) {
        resolve(body as DocumentOut);
      } else {
        reject(toApiError(xhr.status, body, requestId));
      }
    };
    xhr.onerror = () =>
      reject(new ApiError("NETWORK_ERROR", `Could not reach the ClauseLens API at ${baseUrl}.`));
    xhr.onabort = () => reject(new ApiError("ABORTED", "Upload cancelled."));

    const form = new FormData();
    form.append("file", file, file.name);
    xhr.send(form);
  });
  return { promise, abort: () => xhr.abort() };
}
