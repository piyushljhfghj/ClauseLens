import { describe, expect, it } from "vitest";

import {
  INITIAL_PAGE_STATE,
  needsRelease,
  nextPageState,
  pageOverlay,
  type PageRenderEvent,
} from "@/lib/pdfHighlights";

function run(...events: PageRenderEvent[]) {
  return events.reduce(nextPageState, INITIAL_PAGE_STATE);
}

describe("page render state", () => {
  it("shows the placeholder until a page is drawn", () => {
    expect(pageOverlay(INITIAL_PAGE_STATE)).toBe("placeholder");
    expect(pageOverlay(run("drawn"))).toBeNull();
  });

  it("shows a failure even when an older bitmap is still on the canvas", () => {
    const failedRerender = run("drawn", "failed");
    expect(failedRerender.drawn).toBe(true); // the old bitmap is kept underneath...
    expect(pageOverlay(failedRerender)).toBe("error"); // ...but the failure is not hidden
  });

  it("clears a failure once a later render succeeds", () => {
    expect(pageOverlay(run("failed", "drawn"))).toBeNull();
    expect(pageOverlay(run("drawn", "failed", "drawn"))).toBeNull();
  });

  it("returns to the placeholder when a page is released, never to a blank page", () => {
    expect(pageOverlay(run("drawn", "released"))).toBe("placeholder");
    expect(pageOverlay(run("drawn", "failed", "released"))).toBe("placeholder");
  });
});

describe("needsRelease", () => {
  it("releases only rendered pages that left the render area", () => {
    expect(needsRelease(false, true, true)).toBe(true);
  });

  it("never touches pages that were never rendered (no worker requests when a PDF opens)", () => {
    expect(needsRelease(false, true, false)).toBe(false);
  });

  it("does nothing for pages still near the visible area or before the PDF is loaded", () => {
    expect(needsRelease(true, true, true)).toBe(false);
    expect(needsRelease(false, false, true)).toBe(false);
  });
});
