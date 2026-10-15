import { describe, expect, it } from "vitest";

import {
  checkBlockRanges,
  codePointLength,
  codePointToUtf16,
  highlightSegments,
  parseOffsetParam,
  rangesForCanonicalSpan,
  type BlockRange,
  type LayoutBlock,
} from "./citations";

const SMILE = String.fromCodePoint(0x1f600); // 1 code point, 2 UTF-16 units
const ROCKET = String.fromCodePoint(0x1f680);

function layout(...texts: [string, string][]): LayoutBlock[] {
  // Blocks separated by a blank line, as the backend stores them (offsets in code points).
  const blocks: LayoutBlock[] = [];
  let position = 0;
  texts.forEach(([kind, text], ordinal) => {
    if (ordinal) position += 2;
    const length = codePointLength(text);
    blocks.push({ ordinal, kind, char_start: position, char_end: position + length, text });
    position += length;
  });
  return blocks;
}

function marked(text: string, ranges: { start: number; end: number }[]): string {
  const segments = highlightSegments(text, ranges);
  if (!segments) return "<invalid>";
  return segments.map((s) => (s.highlighted ? `[${s.text}]` : s.text)).join("");
}

describe("code points and UTF-16", () => {
  it("counts an emoji as one code point", () => {
    expect(`a${SMILE}b`.length).toBe(4);
    expect(codePointLength(`a${SMILE}b`)).toBe(3);
  });

  it("converts offsets after a surrogate pair", () => {
    const text = `x${SMILE}y${ROCKET}z`;
    expect([0, 1, 2, 3, 4, 5].map((o) => codePointToUtf16(text, o))).toEqual([0, 1, 3, 4, 6, 7]);
  });

  it("rejects invalid offsets", () => {
    expect(codePointToUtf16("abc", 4)).toBeNull();
    expect(codePointToUtf16("abc", -1)).toBeNull();
    expect(codePointToUtf16("abc", 1.5)).toBeNull();
  });
});

describe("highlightSegments", () => {
  it("highlights by code points, never splitting a surrogate pair", () => {
    const text = `Smile ${SMILE} clause applies.`;
    // code points: "Smile " = 0-6, emoji = 6, " " = 7, "clause" = 8-14
    expect(marked(text, [{ start: 6, end: 14 }])).toBe(`Smile [${SMILE} clause] applies.`);
  });

  it("handles several ranges, merging overlaps", () => {
    expect(marked("one two three four", [{ start: 8, end: 13 }, { start: 0, end: 3 }])).toBe(
      "[one] two [three] four",
    );
    expect(marked("one two three", [{ start: 0, end: 5 }, { start: 4, end: 7 }])).toBe("[one two] three");
  });

  it("refuses ranges that do not fit the text", () => {
    expect(highlightSegments("short", [{ start: 2, end: 9 }])).toBeNull();
    expect(highlightSegments("short", [{ start: 3, end: 3 }])).toBeNull();
    expect(highlightSegments("short", [{ start: 4, end: 2 }])).toBeNull();
  });

  it("returns the whole text unhighlighted when there are no ranges", () => {
    expect(highlightSegments("plain", [])).toEqual([{ text: "plain", highlighted: false }]);
  });
});

describe("checkBlockRanges", () => {
  const blocks = layout(
    ["heading", "MASTER AGREEMENT"],
    ["paragraph", `2.1 Either party ${SMILE} may terminate.`],
    ["table_row", "Fee\nAED 100,000"],
  );
  const range = (block: number, start: number, end: number): BlockRange => ({
    block,
    kind: blocks[block].kind,
    block_char_start: blocks[block].char_start,
    block_char_end: blocks[block].char_end,
    start,
    end,
  });

  it("groups valid ranges by block across several blocks", () => {
    const check = checkBlockRanges(blocks, [range(1, 4, 16), range(2, 0, 3)]);
    expect(check.ok).toBe(true);
    if (!check.ok) return;
    expect([...check.byBlock.entries()]).toEqual([
      [1, [{ start: 4, end: 16 }]],
      [2, [{ start: 0, end: 3 }]],
    ]);
    expect(marked(blocks[1].text, check.byBlock.get(1)!)).toBe(`2.1 [Either party] ${SMILE} may terminate.`);
  });

  it("treats a block whose range moved as stale (the document was reprocessed)", () => {
    const moved = { ...range(1, 4, 16), block_char_start: blocks[1].char_start + 5 };
    const check = checkBlockRanges(blocks, [moved]);
    expect(check.ok).toBe(false);
    if (!check.ok) expect(check.reason).toMatch(/reprocessed/);
  });

  it("treats a missing block or an out-of-bounds range as stale", () => {
    expect(checkBlockRanges(blocks, [{ ...range(1, 0, 3), block: 9 }]).ok).toBe(false);
    expect(checkBlockRanges(blocks, [range(2, 0, 99)]).ok).toBe(false);
    expect(checkBlockRanges(blocks, [range(2, 5, 5)]).ok).toBe(false);
  });

  it("refuses everything if any one range is stale", () => {
    expect(checkBlockRanges(blocks, [range(1, 0, 3), { ...range(2, 0, 3), block_char_end: 999 }]).ok).toBe(false);
  });

  it("detects a block whose text does not match its stored range", () => {
    const inconsistent = blocks.map((b) => (b.ordinal === 0 ? { ...b, text: "MASTER" } : b));
    const check = checkBlockRanges(inconsistent, [range(0, 0, 3)]);
    expect(check.ok).toBe(false);
    if (!check.ok) expect(check.reason).toMatch(/inconsistent/);
  });

  it("has nothing to highlight without ranges", () => {
    expect(checkBlockRanges(blocks, []).ok).toBe(false);
  });
});

describe("rangesForCanonicalSpan (link preview only)", () => {
  const blocks = layout(["paragraph", "First block."], ["paragraph", `Second ${SMILE} block.`]);

  it("splits a canonical span across blocks, skipping the separator", () => {
    const ranges = rangesForCanonicalSpan(blocks, 6, blocks[1].char_start + 6);
    expect(ranges?.map((r) => [r.block, r.start, r.end])).toEqual([
      [0, 6, 12],
      [1, 0, 6],
    ]);
  });

  it("rejects empty, reversed or out-of-text spans", () => {
    expect(rangesForCanonicalSpan(blocks, 5, 5)).toBeNull();
    expect(rangesForCanonicalSpan(blocks, 9, 2)).toBeNull();
    expect(rangesForCanonicalSpan(blocks, 500, 600)).toBeNull();
  });
});

describe("parseOffsetParam", () => {
  it("accepts plain non-negative integers only", () => {
    expect(parseOffsetParam("42")).toBe(42);
    expect(parseOffsetParam("0")).toBe(0);
    for (const bad of [null, "", "-1", "1.5", "12abc", " 3", "1e3"]) expect(parseOffsetParam(bad)).toBeNull();
  });
});
