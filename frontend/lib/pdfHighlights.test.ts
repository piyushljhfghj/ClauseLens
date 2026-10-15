import { describe, expect, it } from "vitest";

import {
  checkHighlights,
  overlayPosition,
  parsePageParam,
  parsePreviewHighlights,
  renderScale,
  renderedSizeMatches,
  type HighlightRect,
  type PageSize,
} from "./pdfHighlights";

const A4: PageSize = { page_number: 1, width: 595, height: 842 };
const LANDSCAPE: PageSize = { page_number: 2, width: 842, height: 595 }; // a page rotated 90 degrees
const PAGES = [A4, LANDSCAPE, { page_number: 3, width: 612, height: 792 }];

const rect = (page: PageSize, x0: number, y0: number, x1: number, y1: number): HighlightRect => ({
  page: page.page_number,
  page_width: page.width,
  page_height: page.height,
  x0,
  y0,
  x1,
  y1,
});

describe("overlayPosition", () => {
  it("positions a rectangle as percentages of its page", () => {
    expect(overlayPosition(rect(A4, 59.5, 84.2, 297.5, 105.25))).toEqual({
      left: "10.0000%",
      top: "10.0000%",
      width: "40.0000%",
      height: "2.5000%",
    });
  });

  it("uses the displayed (rotated) page size, so landscape pages need no rotation maths", () => {
    const pos = overlayPosition(rect(LANDSCAPE, 421, 297.5, 842, 595));
    expect(pos).toEqual({ left: "50.0000%", top: "50.0000%", width: "50.0000%", height: "50.0000%" });
  });

  it("clamps rectangles that overhang the page edge", () => {
    const pos = overlayPosition(rect(A4, -0.5, 0, 595.5, 10));
    expect(pos.left).toBe("0.0000%");
    expect(Number.parseFloat(pos.width)).toBeLessThanOrEqual(100);
  });
});

describe("checkHighlights", () => {
  it("groups rectangles by page, including a quote across a page break", () => {
    const check = checkHighlights(
      [rect(A4, 72, 700, 500, 712), rect(A4, 72, 714, 300, 726), rect(LANDSCAPE, 72, 40, 400, 52)],
      PAGES,
    );
    expect(check.ok).toBe(true);
    if (!check.ok) return;
    expect(check.firstPage).toBe(1);
    expect([...check.byPage.entries()].map(([p, r]) => [p, r.length])).toEqual([
      [1, 2],
      [2, 1],
    ]);
  });

  it("starts at the lowest page even if rectangles arrive out of order", () => {
    const check = checkHighlights([rect(LANDSCAPE, 1, 1, 2, 2), rect(A4, 1, 1, 2, 2)], PAGES);
    expect(check.ok && check.firstPage).toBe(1);
  });

  it("has nothing to draw without rectangles", () => {
    const check = checkHighlights([], PAGES);
    expect(check.ok).toBe(false);
    if (!check.ok) expect(check.reason).toMatch(/No position/);
  });

  it("refuses rectangles on a missing page or a page whose size changed", () => {
    expect(checkHighlights([{ ...rect(A4, 1, 1, 2, 2), page: 9 }], PAGES).ok).toBe(false);
    expect(checkHighlights([{ ...rect(A4, 1, 1, 2, 2), page_width: 612 }], PAGES).ok).toBe(false);
    // Within rounding tolerance is fine.
    expect(checkHighlights([{ ...rect(A4, 1, 1, 2, 2), page_width: 595.4 }], PAGES).ok).toBe(true);
  });

  it("refuses empty, inverted, off-page or non-finite rectangles", () => {
    for (const bad of [
      rect(A4, 10, 10, 10, 20),
      rect(A4, 20, 10, 10, 20),
      rect(A4, 10, 10, 700, 20),
      rect(A4, 10, -5, 20, 20),
      { ...rect(A4, 10, 10, 20, 20), x1: Number.NaN },
    ]) {
      expect(checkHighlights([bad], PAGES).ok).toBe(false);
    }
  });

  it("refuses all rectangles if any one is bad", () => {
    expect(checkHighlights([rect(A4, 1, 1, 2, 2), rect(A4, 1, 1, 9999, 2)], PAGES).ok).toBe(false);
  });
});

describe("rendering", () => {
  it("compares PDF.js page sizes with the stored ones", () => {
    expect(renderedSizeMatches(A4, 595.276, 841.89)).toBe(true);
    expect(renderedSizeMatches(A4, 842, 595)).toBe(false); // rendered without the page's rotation
  });

  it("scales to the container width and the screen density, capped", () => {
    expect(renderScale(595, 595, 1)).toBe(1);
    expect(renderScale(297.5, 595, 2)).toBe(1);
    expect(renderScale(595, 595, 4)).toBe(2);
    expect(renderScale(0, 595, 1)).toBe(0);
  });
});

describe("link parameters (preview only)", () => {
  it("parses page numbers", () => {
    expect(parsePageParam("3")).toBe(3);
    for (const bad of [null, "", "0", "-1", "2.5", "3x"]) expect(parsePageParam(bad)).toBeNull();
  });

  it("parses preview rectangles, taking page sizes from the document, not the link", () => {
    expect(parsePreviewHighlights("1:72,100,300,112;2:10.5,20,30,40", PAGES)).toEqual([
      rect(A4, 72, 100, 300, 112),
      rect(LANDSCAPE, 10.5, 20, 30, 40),
    ]);
  });

  it("rejects malformed or unknown-page previews", () => {
    for (const bad of [null, "", "1:72,100,300", "9:1,2,3,4", "1:a,b,c,d", "1:1,2,3,4;"]) {
      expect(parsePreviewHighlights(bad, PAGES)).toBeNull();
    }
  });
});
