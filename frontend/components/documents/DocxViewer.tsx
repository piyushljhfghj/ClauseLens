"use client";

import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, Info } from "lucide-react";
import { Fragment, useEffect, useMemo, useRef } from "react";

import { getLayout } from "@/lib/api/documents";
import {
  checkBlockRanges,
  highlightSegments,
  rangesForCanonicalSpan,
  type BlockRange,
  type LayoutBlock,
  type RangeCheck,
} from "@/lib/citations";

/**
 * What to highlight:
 * - `citation`: the `block_ranges` of a citation the backend verified;
 * - `link`: a canonical [start, end) range from the page URL, a preview aid only. It is shown
 *   as such and never presented as a verified quote.
 */
export type DocxTarget =
  | { kind: "citation"; ranges: BlockRange[] }
  | { kind: "link"; start: number; end: number };

/**
 * A Word document shown as its extracted text: headings, paragraphs and table rows in the
 * order they were stored. This is the text citations were verified against, so a highlight
 * covers exactly the cited characters (Word's own page layout is not reproduced).
 */
export function DocxViewer({
  documentId,
  processedAt,
  target,
}: {
  documentId: string;
  processedAt: string | null;
  target?: DocxTarget;
}) {
  const layout = useQuery({
    queryKey: ["document", documentId, "layout", processedAt],
    queryFn: ({ signal }) => getLayout(documentId, signal),
  });
  const blocks = useMemo(() => layout.data?.blocks ?? [], [layout.data]);
  const check = useMemo(() => resolveTarget(blocks, target), [blocks, target]);
  const firstMark = useRef<HTMLElement | null>(null);
  const firstHighlighted = check?.ok ? Math.min(...check.byBlock.keys()) : null;

  useEffect(() => {
    const mark = firstMark.current;
    if (!mark) return;
    // Instant, not smooth: smooth scrolling was observed not to move the nested scroll area in
    // a mobile viewport, and the highlight itself draws the eye.
    mark.scrollIntoView({ block: "center" });
    mark.focus({ preventScroll: true });
  }, [check]);

  if (layout.isPending) {
    return (
      <div className="space-y-3 rounded-xl border border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-900" aria-busy>
        {[0, 1, 2, 3, 4].map((i) => (
          <div key={i} className="h-4 animate-pulse rounded bg-zinc-100 dark:bg-zinc-800" style={{ width: `${90 - i * 7}%` }} />
        ))}
      </div>
    );
  }
  if (layout.isError) {
    return (
      <div role="alert" className="rounded-xl border border-zinc-200 bg-white px-4 py-12 text-center text-sm dark:border-zinc-800 dark:bg-zinc-900">
        <p className="font-medium">Couldn&apos;t load the document text</p>
        <p className="mt-1 text-zinc-500">{layout.error.message}</p>
      </div>
    );
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <TargetNotice target={target} check={check} />
      <article
        aria-label="Document text"
        className="max-h-[75vh] min-h-[50vh] flex-1 overflow-y-auto rounded-xl border border-zinc-200 bg-white px-5 py-6 text-[15px] leading-7 text-zinc-800 sm:px-8 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-200"
      >
        {blocks.length === 0 ? (
          <p className="text-sm text-zinc-500">No text was extracted from this document.</p>
        ) : (
          blocks.map((block) => {
            const ranges = check?.ok ? check.byBlock.get(block.ordinal) ?? [] : [];
            return (
              <Block
                key={block.ordinal}
                block={block}
                ranges={ranges}
                markRef={block.ordinal === firstHighlighted ? firstMark : undefined}
              />
            );
          })
        )}
      </article>
    </div>
  );
}

function resolveTarget(blocks: LayoutBlock[], target: DocxTarget | undefined): RangeCheck | null {
  if (!target || blocks.length === 0) return null;
  if (target.kind === "citation") return checkBlockRanges(blocks, target.ranges);
  const ranges = rangesForCanonicalSpan(blocks, target.start, target.end);
  return ranges
    ? checkBlockRanges(blocks, ranges)
    : { ok: false, reason: "The linked range is outside this document's text." };
}

function TargetNotice({ target, check }: { target?: DocxTarget; check: RangeCheck | null }) {
  if (!target || !check) return null;
  if (!check.ok) {
    return (
      <p role="status" className="flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-200">
        <AlertTriangle aria-hidden className="mt-0.5 h-4 w-4 shrink-0" />
        {check.reason}
      </p>
    );
  }
  if (target.kind === "link") {
    return (
      <p role="status" className="flex items-start gap-2 rounded-lg border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm text-zinc-600 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-400">
        <Info aria-hidden className="mt-0.5 h-4 w-4 shrink-0" />
        Highlighting characters {target.start}–{target.end} from the link. This is a preview, not a verified citation.
      </p>
    );
  }
  return (
    <p role="status" className="sr-only">
      Cited passage highlighted.
    </p>
  );
}

function Block({
  block,
  ranges,
  markRef,
}: {
  block: LayoutBlock;
  ranges: { start: number; end: number }[];
  markRef?: React.RefObject<HTMLElement | null>;
}) {
  const segments = highlightSegments(block.text, ranges) ?? [{ text: block.text, highlighted: false }];
  const firstMarkIndex = segments.findIndex((s) => s.highlighted);
  const content = segments.map((segment, i) => {
    if (!segment.highlighted) return <Fragment key={i}>{segment.text}</Fragment>;
    const attachRef = markRef !== undefined && i === firstMarkIndex;
    return (
      <mark
        key={i}
        ref={attachRef ? markRef : undefined}
        tabIndex={attachRef ? -1 : undefined}
        className="rounded-sm bg-amber-200 px-0.5 text-inherit outline-none ring-amber-500 focus-visible:ring-2 dark:bg-amber-500/40"
      >
        {segment.text}
      </mark>
    );
  });

  if (block.kind === "heading") {
    return <h3 className="mb-3 mt-6 whitespace-pre-line text-base font-semibold first:mt-0">{content}</h3>;
  }
  if (block.kind === "table_row") {
    // Cells are stored as lines of the row, so they are shown one per line.
    return (
      <div className="mb-2 whitespace-pre-line rounded-md border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-800/50">
        {content}
      </div>
    );
  }
  return <p className="mb-4 whitespace-pre-line">{content}</p>;
}
