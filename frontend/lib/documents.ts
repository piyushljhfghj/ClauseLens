/** Pure helpers for documents: client-side upload checks and display formatting. */

import type { DocumentOut, DocumentStatus, PageOut } from "@/lib/api/documents";

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
  inspecting: "Checking file",
  extracting: "Extracting text",
  ocr: "Running OCR",
  structuring: "Detecting sections",
  embedding: "Building search index",
  saving: "Saving",
};

export function statusLabel(doc: Pick<DocumentOut, "status" | "stage" | "indexed">): string {
  switch (doc.status) {
    case "ready":
      // "Ready" alone would hide the difference between stored-and-checked and searchable.
      return doc.indexed ? "Indexed" : "Not indexed";
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

type CoverageFields = Pick<
  DocumentOut,
  | "file_type"
  | "page_count"
  | "ocr_page_count"
  | "low_quality_page_count"
  | "unreadable_page_count"
  | "section_count"
  | "chunk_count"
>;

/** "150 pages · 12 via OCR · 42 sections · 310 passages" */
export function coverageSummary(doc: CoverageFields): string | null {
  if (doc.chunk_count == null) return null;
  const parts: string[] = [];
  if (doc.file_type === "pdf" && doc.page_count != null) {
    parts.push(`${doc.page_count} ${doc.page_count === 1 ? "page" : "pages"}`);
    if (doc.ocr_page_count) {
      const low = doc.low_quality_page_count ? ` (${doc.low_quality_page_count} low quality)` : "";
      parts.push(`${doc.ocr_page_count} via OCR${low}`);
    }
  }
  if (doc.section_count != null) {
    parts.push(`${doc.section_count} ${doc.section_count === 1 ? "section" : "sections"}`);
  }
  parts.push(`${doc.chunk_count} ${doc.chunk_count === 1 ? "passage" : "passages"}`);
  return parts.join(" · ");
}

/** Honest statement of what was NOT read, or null if everything readable was read. */
export function unreadableNotice(doc: Pick<DocumentOut, "unreadable_page_count" | "page_count">): string | null {
  const n = doc.unreadable_page_count ?? 0;
  if (n === 0) return null;
  const of = doc.page_count != null ? ` of ${doc.page_count}` : "";
  return `${n}${of} ${n === 1 ? "page" : "pages"} could not be read. Answers cannot draw on ${n === 1 ? "it" : "them"}.`;
}

/** Pages read by OCR with low confidence: indexed, but probably only partly recognised. */
export function lowQualityNotice(doc: Pick<DocumentOut, "low_quality_page_count" | "page_count">): string | null {
  const n = doc.low_quality_page_count ?? 0;
  if (n === 0) return null;
  const of = doc.page_count != null ? ` of ${doc.page_count}` : "";
  return `${n}${of} ${n === 1 ? "page was" : "pages were"} read with low OCR quality. Content may be largely unrecognised; answers based on ${n === 1 ? "it" : "them"} may be incomplete.`;
}

export type PageCoverageKind = "text" | "ocr" | "low_quality" | "unreadable" | "blank";

// Mirrors the backend: notes meaning content may exist that was not read.
const UNREADABLE_NOTES = new Set(["ocr_unavailable", "ocr_failed", "ocr_no_text"]);

/** How a page was read, for the coverage strip. Low quality is distinct from unreadable. */
export function pageCoverageKind(page: Pick<PageOut, "extraction" | "note">): PageCoverageKind {
  if (page.note && UNREADABLE_NOTES.has(page.note)) return "unreadable";
  if (page.note === "ocr_low_quality") return "low_quality";
  if (page.extraction === "text" || page.extraction === "ocr") return page.extraction;
  return "blank";
}
