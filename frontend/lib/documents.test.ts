import { describe, expect, it } from "vitest";

import {
  coverageSummary,
  describePages,
  formatBytes,
  isInFlight,
  lowQualityNotice,
  pageCoverageKind,
  statusLabel,
  unreadableNotice,
  validateUploadFile,
} from "./documents";

const MB = 1024 * 1024;

describe("validateUploadFile", () => {
  it.each([
    ["contract.pdf", 1000],
    ["Contract.PDF", 1000],
    ["agreement.docx", 1000],
  ])("accepts %s", (name, size) => {
    expect(validateUploadFile({ name, size })).toBeNull();
  });

  it.each([
    ["notes.txt", 10, /Only PDF/],
    ["scan.png", 10, /Only PDF/],
    ["old.doc", 10, /\.doc files aren't supported/],
    ["empty.pdf", 0, /empty/],
    ["huge.pdf", 26 * MB, /25 MB/],
  ])("rejects %s", (name, size, message) => {
    expect(validateUploadFile({ name, size })).toMatch(message);
  });
});

describe("status helpers", () => {
  it("treats queued and processing as in flight", () => {
    expect(isInFlight("queued")).toBe(true);
    expect(isInFlight("processing")).toBe(true);
    expect(isInFlight("ready")).toBe(false);
    expect(isInFlight("failed")).toBe(false);
  });

  it("labels stages in plain language", () => {
    expect(statusLabel({ status: "processing", stage: "ocr", indexed: false })).toBe("Running OCR");
    expect(statusLabel({ status: "processing", stage: "embedding", indexed: false })).toBe("Building search index");
    expect(statusLabel({ status: "processing", stage: "something-new", indexed: false })).toBe("Processing");
    expect(statusLabel({ status: "queued", stage: "retrying", indexed: false })).toBe("Retrying after an error");
  });

  it("distinguishes stored-and-checked from fully indexed", () => {
    expect(statusLabel({ status: "ready", stage: null, indexed: true })).toBe("Indexed");
    expect(statusLabel({ status: "ready", stage: null, indexed: false })).toBe("Not indexed");
  });
});

describe("formatting", () => {
  it("formats sizes", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(2048)).toBe("2 KB");
    expect(formatBytes(3.5 * MB)).toBe("3.5 MB");
  });

  it("describes pages", () => {
    expect(describePages({ file_type: "pdf", page_count: 1 })).toBe("1 page");
    expect(describePages({ file_type: "pdf", page_count: 150 })).toBe("150 pages");
    expect(describePages({ file_type: "pdf", page_count: null })).toBeNull();
    expect(describePages({ file_type: "docx", page_count: null })).toBe("Word document");
  });
});

describe("coverage", () => {
  const base = {
    file_type: "pdf" as const,
    page_count: 150,
    ocr_page_count: 12,
    unreadable_page_count: 0,
    section_count: 42,
    chunk_count: 310,
  };

  it("summarises what was indexed", () => {
    expect(coverageSummary(base)).toBe("150 pages · 12 via OCR · 42 sections · 310 passages");
    expect(coverageSummary({ ...base, ocr_page_count: 0 })).toBe("150 pages · 42 sections · 310 passages");
    expect(coverageSummary({ ...base, file_type: "docx", page_count: null, ocr_page_count: null })).toBe(
      "42 sections · 310 passages",
    );
    expect(coverageSummary({ ...base, chunk_count: null })).toBeNull();
  });

  it("states unread pages explicitly", () => {
    expect(unreadableNotice(base)).toBeNull();
    expect(unreadableNotice({ ...base, unreadable_page_count: 2 })).toBe(
      "2 of 150 pages could not be read. Answers cannot draw on them.",
    );
    expect(unreadableNotice({ unreadable_page_count: 1, page_count: 3 })).toMatch(/1 of 3 page could not/);
  });
});

describe("low-quality OCR", () => {
  it("adds the low-quality count to the coverage summary", () => {
    const doc = {
      file_type: "pdf" as const,
      page_count: 5,
      ocr_page_count: 4,
      low_quality_page_count: 3,
      unreadable_page_count: 1,
      section_count: 0,
      chunk_count: 1,
    };
    expect(coverageSummary(doc)).toBe("5 pages · 4 via OCR (3 low quality) · 0 sections · 1 passage");
    expect(coverageSummary({ ...doc, low_quality_page_count: 0 })).toBe("5 pages · 4 via OCR · 0 sections · 1 passage");
  });

  it("warns that low-quality pages may be incomplete, separately from unreadable pages", () => {
    expect(lowQualityNotice({ low_quality_page_count: 0, page_count: 5 })).toBeNull();
    expect(lowQualityNotice({ low_quality_page_count: null, page_count: 5 })).toBeNull();
    expect(lowQualityNotice({ low_quality_page_count: 3, page_count: 5 })).toBe(
      "3 of 5 pages were read with low OCR quality. Content may be largely unrecognised; answers based on them may be incomplete.",
    );
    expect(lowQualityNotice({ low_quality_page_count: 1, page_count: 2 })).toMatch(/^1 of 2 page was read/);
  });

  it("classifies pages for the coverage strip, keeping low quality distinct from not read", () => {
    expect(pageCoverageKind({ extraction: "text", note: null })).toBe("text");
    expect(pageCoverageKind({ extraction: "ocr", note: null })).toBe("ocr");
    expect(pageCoverageKind({ extraction: "ocr", note: "ocr_low_quality" })).toBe("low_quality");
    expect(pageCoverageKind({ extraction: "none", note: "ocr_no_text" })).toBe("unreadable");
    expect(pageCoverageKind({ extraction: "none", note: "ocr_failed" })).toBe("unreadable");
    // A stamped scan whose image could not be OCR'd keeps its text layer but was not read.
    expect(pageCoverageKind({ extraction: "text", note: "ocr_unavailable" })).toBe("unreadable");
    expect(pageCoverageKind({ extraction: "none", note: "blank" })).toBe("blank");
  });
});
