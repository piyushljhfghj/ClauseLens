import { describe, expect, it } from "vitest";

import type { Citation } from "@/lib/api/answers";
import { ambiguity, citationLink, isNavigable, type CitationContext, type DocumentVersion } from "@/lib/answers";

const PDF_ID = "11111111-1111-1111-1111-111111111111";
const DOCX_ID = "22222222-2222-2222-2222-222222222222";
const V1 = "2026-10-01T10:00:00.123456Z";
const V2 = "2026-10-02T10:00:00.654321Z";

const pdfDoc: DocumentVersion = { id: PDF_ID, file_type: "pdf", processed_at: V1 };
const docxDoc: DocumentVersion = { id: DOCX_ID, file_type: "docx", processed_at: V1 };

function context(viewed: DocumentVersion, answered: DocumentVersion[] = [viewed]): CitationContext {
  return { viewed, answered };
}

const rect = { page: 2, page_width: 612, page_height: 792, x0: 72, y0: 100, x1: 300, y1: 114 };

function citation(overrides: Partial<Citation> = {}): Citation {
  return {
    n: 1,
    status: "verified",
    proposed_quote: "The Supplier shall deliver",
    cited_evidence_id: "E1",
    evidence_id: "E1",
    document_id: PDF_ID,
    document_filename: "contract.pdf",
    quote: "The Supplier shall deliver",
    char_start: 500,
    char_end: 526,
    page_start: 2,
    page_end: 2,
    highlights: [rect],
    block_ranges: [],
    occurrence_count: 1,
    location_ambiguous: false,
    referenced: true,
    marker_conflict: false,
    reason: null,
    ...overrides,
  };
}

const range = { block: 4, kind: "paragraph", block_char_start: 480, block_char_end: 560, start: 20, end: 46 };

function docxCitation(overrides: Partial<Citation> = {}): Citation {
  return citation({
    document_id: DOCX_ID,
    document_filename: "contract.docx",
    page_start: null,
    page_end: null,
    highlights: [],
    block_ranges: [range],
    ...overrides,
  });
}

describe("citationLink: PDF", () => {
  it("targets the verified rectangles and their first page", () => {
    const second = { ...rect, page: 3, y0: 50, y1: 64 };
    const link = citationLink(citation({ page_end: 3, highlights: [second, rect] }), context(pdfDoc));
    expect(link).toMatchObject({ kind: "pdf", documentId: PDF_ID, page: 2 });
    if (link.kind !== "pdf") throw new Error("expected pdf");
    expect(link.target).toEqual({ kind: "citation", highlights: [second, rect], page: 2 });
    expect(isNavigable(link)).toBe(true);
  });

  it("copies the rectangles rather than sharing the response's objects", () => {
    const source = citation();
    const link = citationLink(source, context(pdfDoc));
    if (link.kind !== "pdf") throw new Error("expected pdf");
    expect(link.target.highlights[0]).not.toBe(source.highlights![0]);
  });

  it("navigates a verified quote without rectangles to its page, explaining why there is no highlight", () => {
    const link = citationLink(
      citation({ highlights: [], page_start: 4, page_end: 5, reason: "Highlight geometry unavailable." }),
      context(pdfDoc),
    );
    expect(link).toMatchObject({ kind: "page", page: 4, target: { kind: "citation", highlights: [], page: 4 } });
    if (link.kind !== "page") throw new Error("expected page");
    expect(link.explanation).toContain("pages 4");
    expect(link.explanation).toContain("exact position");
  });

  it("names a single page in the page-only explanation", () => {
    const link = citationLink(citation({ highlights: [], page_end: null }), context(pdfDoc));
    if (link.kind !== "page") throw new Error("expected page");
    expect(link.explanation).toContain("on page 2,");
  });

  it("refuses a PDF quote with neither rectangles nor a page", () => {
    const link = citationLink(citation({ highlights: [], page_start: null, page_end: null }), context(pdfDoc));
    expect(link).toMatchObject({ kind: "none", code: "no_location" });
  });

  it.each([
    ["a rectangle outside the cited pages", { highlights: [{ ...rect, page: 7 }] }],
    ["an empty rectangle", { highlights: [{ ...rect, x1: rect.x0 }] }],
    ["an inverted rectangle", { highlights: [{ ...rect, y0: 200, y1: 100 }] }],
    ["a non-finite coordinate", { highlights: [{ ...rect, x1: Number.NaN }] }],
    ["a zero page size", { highlights: [{ ...rect, page_width: 0 }] }],
    ["a fractional page number", { highlights: [{ ...rect, page: 2.5 }] }],
    ["rectangles without a page range", { page_start: null, page_end: null }],
    ["a page range that runs backwards", { page_start: 3, page_end: 2 }],
    ["a page-only quote on page 0", { highlights: [], page_start: 0, page_end: 0 }],
  ] as const)("refuses %s", (_label, overrides) => {
    const link = citationLink(citation(overrides as Partial<Citation>), context(pdfDoc));
    expect(link).toMatchObject({ kind: "none", code: "invalid_location" });
    expect(isNavigable(link)).toBe(false);
  });

  it("refuses Word block ranges on a PDF", () => {
    const link = citationLink(citation({ block_ranges: [range] }), context(pdfDoc));
    expect(link).toMatchObject({ kind: "none", code: "file_type_mismatch" });
  });
});

describe("citationLink: Word", () => {
  it("targets the verified block ranges", () => {
    const link = citationLink(docxCitation(), context(docxDoc));
    expect(link).toMatchObject({ kind: "docx", documentId: DOCX_ID });
    if (link.kind !== "docx") throw new Error("expected docx");
    expect(link.target).toEqual({ kind: "citation", ranges: [range] });
  });

  it("accepts a quote spanning several blocks in order", () => {
    const first = { ...range, start: 20, end: 80 };
    const second = { block: 5, kind: "table_row", block_char_start: 562, block_char_end: 600, start: 0, end: 10 };
    const link = citationLink(docxCitation({ char_end: 572, block_ranges: [first, second] }), context(docxDoc));
    expect(link.kind).toBe("docx");
  });

  it("refuses a Word quote with no ranges (Word has no pages to fall back to)", () => {
    const link = citationLink(docxCitation({ block_ranges: [], page_start: 1 }), context(docxDoc));
    expect(link).toMatchObject({ kind: "none", code: "no_location" });
  });

  it.each([
    ["a range past the end of its block", [{ ...range, end: 81 }]],
    ["an empty range", [{ ...range, end: range.start }]],
    ["a negative block", [{ ...range, block: -1 }]],
    ["a range outside the quote's span", [{ ...range, start: 0, end: 10 }]],
    ["blocks out of order", [{ ...range, block: 5 }, { ...range, block: 4, block_char_start: 562, block_char_end: 600, start: 0, end: 2 }]],
    ["overlapping blocks", [{ ...range, end: 80 }, { ...range, block: 5, block_char_start: 550, start: 0, end: 2 }]],
  ] as const)("refuses %s", (_label, ranges) => {
    const link = citationLink(docxCitation({ char_end: 600, block_ranges: [...ranges] }), context(docxDoc));
    expect(link).toMatchObject({ kind: "none", code: "invalid_location" });
  });

  it("refuses ranges when the quote's canonical span is missing", () => {
    const link = citationLink(docxCitation({ char_start: null }), context(docxDoc));
    expect(link).toMatchObject({ kind: "none", code: "invalid_location" });
  });

  it("refuses PDF rectangles on a Word document", () => {
    const link = citationLink(docxCitation({ highlights: [rect] }), context(docxDoc));
    expect(link).toMatchObject({ kind: "none", code: "file_type_mismatch" });
  });
});

describe("citationLink: refusals before any location is used", () => {
  it("never makes an unverified quote clickable, even if it carries location data", () => {
    const link = citationLink(citation({ status: "unverified", reason: "Quote not found." }), context(pdfDoc));
    expect(link).toMatchObject({ kind: "none", code: "unverified", serverReason: "Quote not found." });
  });

  it("refuses a verified quote without a document", () => {
    expect(citationLink(citation({ document_id: null }), context(pdfDoc))).toMatchObject({ code: "invalid_location" });
  });

  it("refuses a quote from another document", () => {
    const link = citationLink(docxCitation(), context(pdfDoc, [pdfDoc, docxDoc]));
    expect(link).toMatchObject({ kind: "none", code: "other_document" });
  });

  it("refuses a quote when the document was reprocessed after the answer", () => {
    const link = citationLink(citation(), context({ ...pdfDoc, processed_at: V2 }, [pdfDoc]));
    expect(link).toMatchObject({ kind: "none", code: "stale_version" });
  });

  it("compares versions exactly (microseconds included)", () => {
    const close = "2026-10-01T10:00:00.123457Z";
    const link = citationLink(citation(), context({ ...pdfDoc, processed_at: close }, [pdfDoc]));
    expect(link).toMatchObject({ code: "stale_version" });
  });

  it.each([
    ["the answer does not record the document", [] as DocumentVersion[], pdfDoc],
    ["the answer's version is null", [{ ...pdfDoc, processed_at: null }], pdfDoc],
    ["the viewed document has no index", [pdfDoc], { ...pdfDoc, processed_at: null }],
    ["the answer records the document twice", [pdfDoc, pdfDoc], pdfDoc],
  ])("refuses a quote when %s", (_label, answered, viewed) => {
    const link = citationLink(citation(), context(viewed, answered));
    expect(link).toMatchObject({ kind: "none", code: "version_unknown" });
  });

  it("refuses when the answer's file type differs from the viewed document's", () => {
    const link = citationLink(citation(), context(pdfDoc, [{ ...pdfDoc, file_type: "docx" }]));
    expect(link).toMatchObject({ kind: "none", code: "file_type_mismatch" });
  });

  it("passes the backend's reason through as plain text without interpreting it", () => {
    const reason = "<img src=x onerror=alert(1)>";
    const link = citationLink(citation({ status: "unverified", reason }), context(pdfDoc));
    if (link.kind !== "none") throw new Error("expected none");
    expect(link.serverReason).toBe(reason);
    expect(link.reason).not.toContain("<");
  });
});

describe("ambiguity", () => {
  it("never opens an ambiguous PDF quote at one occurrence, and says why", () => {
    const link = citationLink(citation({ location_ambiguous: true, occurrence_count: 3 }), context(pdfDoc));
    expect(link).toMatchObject({ kind: "none", code: "ambiguous_location" });
    expect(isNavigable(link)).toBe(false);
    if (link.kind !== "none") throw new Error("expected none");
    expect(link.reason).toContain("3 times");
    expect(link.reason).toContain("isn't opened");
  });

  it("never opens an ambiguous Word quote or page-only PDF quote either", () => {
    const docx = citation({
      document_id: DOCX_ID, page_start: null, page_end: null, highlights: [], block_ranges: [range],
      location_ambiguous: true, occurrence_count: 2,
    });  // prettier-ignore
    expect(citationLink(docx, context(docxDoc))).toMatchObject({ code: "ambiguous_location" });
    const pageOnly = citation({ highlights: [], location_ambiguous: true, occurrence_count: 2 });
    expect(citationLink(pageOnly, context(pdfDoc))).toMatchObject({ code: "ambiguous_location" });
  });

  it("reports stale or other-document problems before ambiguity", () => {
    const ambiguous = citation({ location_ambiguous: true, occurrence_count: 2 });
    expect(citationLink(ambiguous, context({ ...pdfDoc, processed_at: V2 }, [pdfDoc]))).toMatchObject({
      code: "stale_version",
    });
  });

  it("opens a repeated quote whose location the backend could determine", () => {
    // occurrence_count > 1 alone is not ambiguity: the verifier located the cited occurrence.
    const link = citationLink(citation({ occurrence_count: 3, location_ambiguous: false }), context(pdfDoc));
    expect(link.kind).toBe("pdf");
  });

  it("still describes ambiguity when the count is missing", () => {
    expect(ambiguity(citation({ location_ambiguous: true, occurrence_count: null }))).toMatchObject({ occurrences: null });
  });

  it("is null for an unambiguous quote, even one that repeats elsewhere", () => {
    expect(ambiguity(citation({ occurrence_count: 4 }))).toBeNull();
  });

  it("is null for an unverified quote", () => {
    expect(ambiguity(citation({ status: "unverified", location_ambiguous: true }))).toBeNull();
  });
});
