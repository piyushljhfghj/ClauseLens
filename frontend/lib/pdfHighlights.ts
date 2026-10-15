/**
 * Pure helpers for drawing verified citation rectangles over rendered PDF pages.
 *
 * The API gives each rectangle in PDF points on the *displayed* page (rotation already applied,
 * origin top-left, y increasing downwards) together with that page's displayed size. Overlays
 * are therefore positioned as percentages of the page, which stays aligned at any zoom or width
 * without any rotation maths, as long as the page is rendered in its displayed orientation.
 *
 * Nothing here makes a location trustworthy: rectangles come from the backend's verified
 * citations. These helpers only refuse to draw rectangles that do not fit the pages shown.
 */

import type { components } from "@/lib/api/schema";

export type HighlightRect = components["schemas"]["HighlightRect"];

export interface PageSize {
  page_number: number;
  width: number;
  height: number;
}

/** Stored and rendered page sizes are rounded to 0.01 pt; anything larger is a real mismatch. */
export const SIZE_TOLERANCE_PT = 1;

export interface OverlayPosition {
  left: string;
  top: string;
  width: string;
  height: string;
}

function percent(value: number): string {
  return `${Math.min(100, Math.max(0, value * 100)).toFixed(4)}%`;
}

/** CSS position of `rect` as percentages of its page, clamped to the page. */
export function overlayPosition(rect: HighlightRect): OverlayPosition {
  const left = rect.x0 / rect.page_width;
  const top = rect.y0 / rect.page_height;
  return {
    left: percent(left),
    top: percent(top),
    width: percent(rect.x1 / rect.page_width - left),
    height: percent(rect.y1 / rect.page_height - top),
  };
}

export type HighlightCheck =
  | { ok: true; byPage: Map<number, HighlightRect[]>; firstPage: number }
  | { ok: false; reason: string };

const INCONSISTENT =
  "This document's page layout doesn't match the location (it may have been reprocessed), so " +
  "the passage can't be highlighted reliably.";

/**
 * Check `highlights` against the document's pages and group them by page. Every rectangle must
 * be on an existing page whose size matches the size recorded with the rectangle, and lie inside
 * it; otherwise nothing is drawn.
 */
export function checkHighlights(highlights: HighlightRect[], pages: PageSize[]): HighlightCheck {
  if (highlights.length === 0) {
    return { ok: false, reason: "No position on the page is available for this quote." };
  }
  const byNumber = new Map(pages.map((p) => [p.page_number, p]));
  const byPage = new Map<number, HighlightRect[]>();
  for (const rect of highlights) {
    const values = [rect.page_width, rect.page_height, rect.x0, rect.y0, rect.x1, rect.y1];
    if (!values.every(Number.isFinite) || !Number.isInteger(rect.page)) return { ok: false, reason: INCONSISTENT };
    const page = byNumber.get(rect.page);
    if (
      !page ||
      Math.abs(page.width - rect.page_width) > SIZE_TOLERANCE_PT ||
      Math.abs(page.height - rect.page_height) > SIZE_TOLERANCE_PT
    ) {
      return { ok: false, reason: INCONSISTENT };
    }
    const t = SIZE_TOLERANCE_PT;
    if (
      rect.x1 <= rect.x0 ||
      rect.y1 <= rect.y0 ||
      rect.x0 < -t ||
      rect.y0 < -t ||
      rect.x1 > rect.page_width + t ||
      rect.y1 > rect.page_height + t
    ) {
      return { ok: false, reason: INCONSISTENT };
    }
    byPage.set(rect.page, [...(byPage.get(rect.page) ?? []), rect]);
  }
  return { ok: true, byPage, firstPage: Math.min(...byPage.keys()) };
}

/** Whether the size PDF.js reports for a page (scale 1, displayed orientation) matches the size
 * recorded at extraction; if not, overlays on that page would be misplaced. */
export function renderedSizeMatches(page: PageSize, width: number, height: number): boolean {
  return Math.abs(page.width - width) <= SIZE_TOLERANCE_PT && Math.abs(page.height - height) <= SIZE_TOLERANCE_PT;
}

/** The PDF.js scale for rendering a page `pageWidthPt` wide into `cssWidth` CSS pixels, sharp on
 * high-density screens (device pixel ratio capped to bound canvas memory). */
export function renderScale(cssWidth: number, pageWidthPt: number, devicePixelRatio: number): number {
  if (!(cssWidth > 0) || !(pageWidthPt > 0)) return 0;
  const ratio = Math.min(Math.max(devicePixelRatio || 1, 1), 2);
  return (cssWidth / pageWidthPt) * ratio;
}

/** A page number (>= 1) from a URL parameter, or null. */
export function parsePageParam(value: string | null): number | null {
  if (value === null || !/^[1-9]\d{0,5}$/.test(value)) return null;
  return Number(value);
}

/**
 * Preview rectangles from a link (`hl=<page>:<x0>,<y0>,<x1>,<y1>;...`), for checking the viewer
 * before citations can be opened from answers. Page sizes are taken from the document's pages,
 * never from the link, and the result is only ever shown as a preview, not as a citation.
 * Returns null for anything malformed.
 */
export function parsePreviewHighlights(value: string | null, pages: PageSize[]): HighlightRect[] | null {
  if (!value) return null;
  const byNumber = new Map(pages.map((p) => [p.page_number, p]));
  const rects: HighlightRect[] = [];
  for (const part of value.split(";")) {
    const match = /^([1-9]\d{0,5}):(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)$/.exec(part);
    if (!match) return null;
    const page = byNumber.get(Number(match[1]));
    if (!page) return null;
    const [x0, y0, x1, y1] = match.slice(2).map(Number);
    rects.push({ page: page.page_number, page_width: page.width, page_height: page.height, x0, y0, x1, y1 });
  }
  return rects.length > 0 && rects.length <= 50 ? rects : null;
}

/** Rendering state of one page: whether its canvas holds a drawn bitmap, and whether the last
 * render attempt failed. A failure stays visible even over an older bitmap. */
export interface PageRenderState {
  drawn: boolean;
  failed: boolean;
}

export type PageRenderEvent = "drawn" | "failed" | "released";

export const INITIAL_PAGE_STATE: PageRenderState = { drawn: false, failed: false };

export function nextPageState(state: PageRenderState, event: PageRenderEvent): PageRenderState {
  if (event === "drawn") return { drawn: true, failed: false };
  if (event === "failed") return { drawn: state.drawn, failed: true };
  return INITIAL_PAGE_STATE; // released: the canvas was cleared
}

/** What to show over the canvas: the "Page N" placeholder while nothing is drawn, an error if
 * the last render failed (even if an older, differently sized bitmap is still underneath). */
export function pageOverlay(state: PageRenderState): "placeholder" | "error" | null {
  if (state.failed) return "error";
  return state.drawn ? null : "placeholder";
}

/** Whether a page leaving the render area has anything to release: only pages that requested
 * their PDF.js page object (to render it) ever hold its resources. */
export function needsRelease(near: boolean, pdfReady: boolean, touched: boolean): boolean {
  return !near && pdfReady && touched;
}
