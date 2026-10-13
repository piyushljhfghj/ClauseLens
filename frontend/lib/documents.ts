/** Pure helpers for documents: client-side upload checks and display formatting. */

import type { DocumentOut, DocumentStatus } from "@/lib/api/documents";

/** Mirrors the backend default (MAX_UPLOAD_MB). The backend remains the authority; this check
 * only saves the user from waiting on an upload that will be rejected. */
export const MAX_UPLOAD_MB = 25;
export const ACCEPTED_EXTENSIONS = [".pdf", ".docx"] as const;
export const ACCEPT_ATTRIBUTE =
  ".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document";

export function validateUploadFile(file: { name: string; size: number }): string | null {
  const name = file.name.toLowerCase();
  if (name.endsWith(".doc")) {
    return "Legacy Word .doc files aren't supported. Save it as .docx in Word and upload that.";
  }
  if (!ACCEPTED_EXTENSIONS.some((ext) => name.endsWith(ext))) {
    return "Unsupported file type. Only PDF (.pdf) and Word (.docx) files are supported.";
  }
  if (file.size === 0) return "The file is empty.";
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
    return `The file is larger than the ${MAX_UPLOAD_MB} MB limit.`;
  }
  return null;
}

export function isInFlight(status: DocumentStatus): boolean {
  return status === "queued" || status === "processing";
}

const STAGE_LABELS: Record<string, string> = {
  waiting: "Waiting to start",
  starting: "Starting",
  retrying: "Retrying after an error",
  downloading: "Preparing file",
  inspecting: "Reading pages",
};

export function statusLabel(doc: Pick<DocumentOut, "status" | "stage">): string {
  switch (doc.status) {
    case "ready":
      return "Ready";
    case "failed":
      return "Failed";
    case "queued":
      return doc.stage === "retrying" ? STAGE_LABELS.retrying : "Queued";
    case "processing":
      return (doc.stage && STAGE_LABELS[doc.stage]) || "Processing";
  }
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function formatDateTime(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function describePages(doc: Pick<DocumentOut, "file_type" | "page_count">): string | null {
  if (doc.file_type === "docx") return "Word document";
  if (doc.page_count == null) return null;
  return `${doc.page_count} ${doc.page_count === 1 ? "page" : "pages"}`;
}
