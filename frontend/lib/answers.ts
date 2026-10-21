/**
 * What clicking a citation in an answer may do in the document viewer.
 *
 * A citation becomes navigable only if all of these hold; otherwise it is shown but cannot be
 * clicked, with a reason:
 * - the backend verified the quote (an unverified one has no trustworthy location at all);
 * - it belongs to the document being viewed;
 * - the answer was produced from the same version of that document as the one on screen, so
 *   offsets, page numbers and rectangles still refer to the text the viewer shows;
 * - the location is unambiguous: a quote whose wording occurs several times, with no way to tell
 *   which occurrence was meant, is shown with an explanation but never opened at one of them;
 * - its location data fits the document's file type and is well formed.
 *
 * The viewers check the location again against the pages or blocks they load, so this is a
 * first line of defence, not the only one. Every string returned here is plain text, to be
 * rendered as text (never as HTML); the backend's `reason` is passed through unchanged.
 */

import type { DocxTarget } from "@/components/documents/DocxViewer";
import type { PdfTarget } from "@/components/documents/PdfViewer";
import type { Citation } from "@/lib/api/answers";
import type { BlockRange } from "@/lib/citations";
import type { HighlightRect } from "@/lib/pdfHighlights";

/** One version of a document: the viewed one, or one an answer was produced from. */
export interface DocumentVersion {
  id: string;
  file_type: "pdf" | "docx";
  /** When its index was built; null if it has none (citations can't refer to it). */
  processed_at: string | null;
}

export interface CitationContext {
  /** The document shown in the viewer. */
  viewed: DocumentVersion;
  /** The document versions the answer was produced from. */
  answered: readonly DocumentVersion[];
}

export type PdfCitationTarget = Extract<PdfTarget, { kind: "citation" }>;
export type DocxCitationTarget = Extract<DocxTarget, { kind: "citation" }>;

/** The quote occurs more than once and the backend could not tell which occurrence was meant. */
export interface Ambiguity {
  occurrences: number | null;
  note: string;
}

export type NotNavigableCode =
  | "unverified"
  | "other_document"
  | "version_unknown"
  | "stale_version"
  | "file_type_mismatch"
  | "ambiguous_location"
  | "no_location"
  | "invalid_location";

export type CitationLink =
  /** Verified PDF quote with rectangles on its page(s). */
  | { kind: "pdf"; documentId: string; page: number; target: PdfCitationTarget }
  /** Verified PDF quote whose page is known but not its exact position on the page. */
  | {
      kind: "page";
      documentId: string;
      page: number;
      target: PdfCitationTarget;
      explanation: string;
    }
  /** Verified Word quote with ranges in the extracted text. */
  | { kind: "docx"; documentId: string; target: DocxCitationTarget }
  | { kind: "none"; code: NotNavigableCode; reason: string; serverReason: string | null };

function none(code: NotNavigableCode, reason: string, citation: Citation): CitationLink {
  return { kind: "none", code, reason, serverReason: citation.reason ?? null };
}

const isIndex = (n: unknown): n is number => typeof n === "number" && Number.isInteger(n) && n >= 0;
const isPage = (n: unknown): n is number => isIndex(n) && n >= 1;
const isPositive = (n: unknown): n is number => typeof n === "number" && Number.isFinite(n) && n > 0;
const isFiniteNumber = (n: unknown): n is number => typeof n === "number" && Number.isFinite(n);

/** The page range a citation reports, or null if absent or malformed. */
function pageRange(citation: Citation): { start: number; end: number } | null {
  const start = citation.page_start;
  const end = citation.page_end ?? start;
  if (!isPage(start) || !isPage(end) || end < start) return null;
  return { start, end };
}

function validRect(rect: HighlightRect, pages: { start: number; end: number }): boolean {
  return (
    isPage(rect.page) &&
    rect.page >= pages.start &&
    rect.page <= pages.end &&
    isPositive(rect.page_width) &&
    isPositive(rect.page_height) &&
    [rect.x0, rect.y0, rect.x1, rect.y1].every(isFiniteNumber) &&
    rect.x1 > rect.x0 &&
    rect.y1 > rect.y0
  );
}

/** Ranges in increasing block order, each inside its block and inside the quote's span. */
function validRanges(ranges: BlockRange[], charStart: number, charEnd: number): boolean {
  let previousBlock = -1;
  let previousEnd = -1;
  for (const r of ranges) {
    if (
      ![r.block, r.block_char_start, r.block_char_end, r.start, r.end].every(isIndex) ||
      r.block <= previousBlock ||
      r.block_char_start < previousEnd ||
      r.block_char_end <= r.block_char_start ||
      r.end <= r.start ||
      r.end > r.block_char_end - r.block_char_start ||
      r.block_char_start + r.start < charStart ||
      r.block_char_start + r.end > charEnd
    ) {
      return false;
    }
    previousBlock = r.block;
    previousEnd = r.block_char_end;
  }
  return true;
}

export function ambiguity(citation: Citation): Ambiguity | null {
  if (citation.status !== "verified" || !citation.location_ambiguous) return null;
  const count = isPage(citation.occurrence_count) && citation.occurrence_count > 1 ? citation.occurrence_count : null;
  return {
    occurrences: count,
    note:
      `This exact wording appears ${count ? `${count} times` : "more than once"} in the document, ` +
      "and it can't be determined which occurrence the answer relied on, so it isn't opened at any " +
      "one of them.",
  };
}

const INVALID = "The location recorded for this quote is malformed, so it can't be shown.";

/** What clicking `citation` may do in the viewer showing `context.viewed`. */
export function citationLink(citation: Citation, context: CitationContext): CitationLink {
  if (citation.status !== "verified") {
    return none("unverified", "This quote could not be verified in the document, so it has no location.", citation);
  }
  const documentId = citation.document_id;
  if (!documentId) return none("invalid_location", INVALID, citation);
  const { viewed } = context;
  if (documentId !== viewed.id) {
    return none("other_document", "This quote is from a different document than the one shown.", citation);
  }

  const answered = context.answered.filter((d) => d.id === documentId);
  if (answered.length !== 1 || answered[0].processed_at === null || viewed.processed_at === null) {
    return none(
      "version_unknown",
      "It isn't known which version of the document this answer used, so the quote can't be located safely.",
      citation,
    );
  }
  if (answered[0].processed_at !== viewed.processed_at) {
    return none(
      "stale_version",
      "The document has been reprocessed since this answer was produced, so the quote's location may no longer be accurate.",
      citation,
    );
  }
  if (answered[0].file_type !== viewed.file_type) {
    return none("file_type_mismatch", "The answer's record of this document's file type doesn't match.", citation);
  }

  const ambiguous = ambiguity(citation);
  if (ambiguous) return none("ambiguous_location", ambiguous.note, citation);

  const highlights = citation.highlights ?? [];
  const ranges = citation.block_ranges ?? [];

  if (viewed.file_type === "pdf") {
    if (ranges.length > 0) {
      return none("file_type_mismatch", "This quote has a Word-document location but the document is a PDF.", citation);
    }
    const pages = pageRange(citation);
    if (highlights.length > 0) {
      if (!pages || !highlights.every((rect) => validRect(rect, pages))) {
        return none("invalid_location", INVALID, citation);
      }
      const page = Math.min(...highlights.map((r) => r.page));
      return {
        kind: "pdf",
        documentId,
        page,
        target: { kind: "citation", highlights: highlights.map((r) => ({ ...r })), page },
      };
    }
    if (citation.page_start === null || citation.page_start === undefined) {
      return none("no_location", "No page or position is available for this quote.", citation);
    }
    if (!pages) return none("invalid_location", INVALID, citation);
    const span = pages.end > pages.start ? `pages ${pages.start}–${pages.end}` : `page ${pages.start}`;
    return {
      kind: "page",
      documentId,
      page: pages.start,
      target: { kind: "citation", highlights: [], page: pages.start },
      explanation: `The quote is on ${span}, but its exact position on the page isn't available, so it can't be highlighted.`,
    };
  }

  if (highlights.length > 0) {
    return none("file_type_mismatch", "This quote has a PDF page location but the document is a Word file.", citation);
  }
  if (ranges.length === 0) {
    return none("no_location", "No location in the text is available for this quote.", citation);
  }
  const { char_start: start, char_end: end } = citation;
  if (!isIndex(start) || !isIndex(end) || end <= start || !validRanges(ranges, start, end)) {
    return none("invalid_location", INVALID, citation);
  }
  return {
    kind: "docx",
    documentId,
    target: { kind: "citation", ranges: ranges.map((r) => ({ ...r })) },
  };
}

/** Whether clicking the citation opens a location in the viewer. */
export function isNavigable(link: CitationLink): link is Exclude<CitationLink, { kind: "none" }> {
  return link.kind !== "none";
}
