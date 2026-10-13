import { describe, expect, it } from "vitest";

import { describePages, formatBytes, isInFlight, statusLabel, validateUploadFile } from "./documents";

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
    expect(statusLabel({ status: "processing", stage: "inspecting" })).toBe("Reading pages");
    expect(statusLabel({ status: "processing", stage: "something-new" })).toBe("Processing");
    expect(statusLabel({ status: "queued", stage: "retrying" })).toBe("Retrying after an error");
    expect(statusLabel({ status: "ready", stage: null })).toBe("Ready");
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
