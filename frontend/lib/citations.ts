/**
 * Pure helpers for showing citation locations in the extracted text of a Word document.
 *
 * The API counts character offsets in Unicode code points (as Python does). JavaScript strings
 * index UTF-16 code units, which differ for characters outside the Basic Multilingual Plane
 * (an emoji is 1 code point but 2 units), so offsets are converted before slicing.
 *
 * Nothing here decides whether a quote is genuine: that is the backend's verification. These
 * helpers only check that a location still fits the text it is shown on, and refuse to
 * highlight anything when it does not.
 */

import type { components } from "@/lib/api/schema";

export type LayoutBlock = components["schemas"]["LayoutBlock"];
export type BlockRange = components["schemas"]["BlockRange"];

/** A [start, end) range of code points within one block's text. */
export interface InBlockRange {
  start: number;
  end: number;
}

export interface TextSegment {
  text: string;
  highlighted: boolean;
}

/** Number of code points in `text` (not `text.length`, which counts UTF-16 units). */
export function codePointLength(text: string): number {
  return Array.from(text).length;
}

/** The UTF-16 index of code-point offset `offset` in `text`, or null if it is not a valid
 * offset (negative, fractional, or past the end). `offset === length` is the end of the text. */
export function codePointToUtf16(text: string, offset: number): number | null {
  if (!Number.isInteger(offset) || offset < 0) return null;
  let points = 0;
  let units = 0;
  for (const ch of text) {
    if (points === offset) return units;
    points += 1;
    units += ch.length;
  }
  return points === offset ? units : null;
}

/**
 * Split `text` into plain and highlighted segments for code-point `ranges`. Overlapping or
 * touching ranges are merged. Returns null if any range is invalid for this text (empty,
 * reversed, or outside it), so a bad location is never drawn over the wrong words.
 */
export function highlightSegments(text: string, ranges: InBlockRange[]): TextSegment[] | null {
  const spans: [number, number][] = [];
  for (const { start, end } of ranges) {
    if (!(start < end)) return null;
    const from = codePointToUtf16(text, start);
    const to = codePointToUtf16(text, end);
    if (from === null || to === null) return null;
    spans.push([from, to]);
  }
  spans.sort((a, b) => a[0] - b[0]);
  const merged: [number, number][] = [];
  for (const span of spans) {
    const last = merged[merged.length - 1];
    if (last && span[0] <= last[1]) last[1] = Math.max(last[1], span[1]);
    else merged.push([span[0], span[1]]);
  }
  const segments: TextSegment[] = [];
  let position = 0;
  for (const [from, to] of merged) {
    if (from > position) segments.push({ text: text.slice(position, from), highlighted: false });
    segments.push({ text: text.slice(from, to), highlighted: true });
    position = to;
  }
  if (position < text.length) segments.push({ text: text.slice(position), highlighted: false });
  return segments;
}

export type RangeCheck =
  | { ok: true; byBlock: Map<number, InBlockRange[]> }
  | { ok: false; reason: string };

const STALE =
  "The document has changed since this location was computed (it was reprocessed), so it " +
  "can't be highlighted reliably.";

/**
 * Check citation `ranges` against the blocks currently shown and group them by block. Every
 * range must name an existing block with the same canonical range it had when the citation was
 * verified, and fit inside that block's text; otherwise nothing is highlighted.
 */
export function checkBlockRanges(blocks: LayoutBlock[], ranges: BlockRange[]): RangeCheck {
  if (ranges.length === 0) return { ok: false, reason: "No location in the text is available for this quote." };
  const byOrdinal = new Map(blocks.map((b) => [b.ordinal, b]));
  const byBlock = new Map<number, InBlockRange[]>();
  for (const range of ranges) {
    const block = byOrdinal.get(range.block);
    if (!block || block.char_start !== range.block_char_start || block.char_end !== range.block_char_end) {
      return { ok: false, reason: STALE };
    }
    const length = codePointLength(block.text);
    if (length !== block.char_end - block.char_start) {
      return { ok: false, reason: "The stored text of this document is inconsistent, so the location can't be shown." };
    }
    if (!Number.isInteger(range.start) || !Number.isInteger(range.end) || range.start < 0 || range.end > length || range.start >= range.end) {
      return { ok: false, reason: STALE };
    }
    byBlock.set(range.block, [...(byBlock.get(range.block) ?? []), { start: range.start, end: range.end }]);
  }
  return { ok: true, byBlock };
}

/**
 * Block ranges for a canonical [start, end) range, computed from the blocks themselves. Used
 * only for a link-supplied preview highlight, never as a verified citation location. Returns
 * null if the range is invalid or touches no block.
 */
export function rangesForCanonicalSpan(blocks: LayoutBlock[], start: number, end: number): BlockRange[] | null {
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || start >= end) return null;
  const ranges: BlockRange[] = [];
  for (const block of [...blocks].sort((a, b) => a.ordinal - b.ordinal)) {
    const from = Math.max(start, block.char_start);
    const to = Math.min(end, block.char_end);
    if (from < to) {
      ranges.push({
        block: block.ordinal,
        kind: block.kind,
        block_char_start: block.char_start,
        block_char_end: block.char_end,
        start: from - block.char_start,
        end: to - block.char_start,
      });
    }
  }
  return ranges.length > 0 ? ranges : null;
}

/** A non-negative integer from a URL parameter, or null (no partial parses like "12abc"). */
export function parseOffsetParam(value: string | null): number | null {
  if (value === null || !/^\d{1,9}$/.test(value)) return null;
  return Number(value);
}
