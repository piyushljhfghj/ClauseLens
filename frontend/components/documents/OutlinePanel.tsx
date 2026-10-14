"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { getOutline, type DocumentOut } from "@/lib/api/documents";

const INITIAL_VISIBLE = 40;

/** The clause tree detected from numbering and headings. */
export function OutlinePanel({ document }: { document: DocumentOut }) {
  const [showAll, setShowAll] = useState(false);
  const outline = useQuery({
    queryKey: ["document", document.id, "outline", document.processed_at],
    queryFn: ({ signal }) => getOutline(document.id, signal),
    enabled: document.indexed,
  });

  const sections = outline.data ?? [];
  const visible = showAll ? sections : sections.slice(0, INITIAL_VISIBLE);

  return (
    <section aria-labelledby="outline-heading" className="rounded-xl border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
      <h2 id="outline-heading" className="text-sm font-semibold">
        Structure
      </h2>
      {outline.isPending ? (
        <div className="mt-3 space-y-2">
          {[0, 1, 2, 3].map((i) => (
            <div key={i} className="h-4 animate-pulse rounded bg-zinc-100 dark:bg-zinc-800" />
          ))}
        </div>
      ) : outline.isError ? (
        <p className="mt-2 text-xs text-red-600">Couldn&apos;t load the structure: {outline.error.message}</p>
      ) : sections.length === 0 ? (
        <p className="mt-2 text-xs text-zinc-500">
          No numbered clauses or headings were detected. The document is still fully searchable as passages.
        </p>
      ) : (
        <>
          <ol className="mt-2 max-h-[50vh] space-y-0.5 overflow-y-auto pr-1 text-sm">
            {visible.map((section) => (
              <li
                key={section.id}
                title={section.title ?? section.preview}
                className="flex items-baseline justify-between gap-3 rounded px-1 py-0.5 hover:bg-zinc-50 dark:hover:bg-zinc-800/60"
                style={{ paddingLeft: `${(section.level - 1) * 14 + 4}px` }}
              >
                <span className="min-w-0 truncate">
                  {section.number ? <span className="font-medium tabular-nums">{section.number} </span> : null}
                  {section.title ? (
                    <span>{section.title}</span>
                  ) : (
                    // No heading text: show the section's opening words rather than a bare number.
                    <span className="text-zinc-500 italic dark:text-zinc-400">{section.preview || "Untitled"}</span>
                  )}
                </span>
                {section.page_start != null ? (
                  <span className="shrink-0 text-xs tabular-nums text-zinc-400">
                    p. {section.page_start}
                    {section.page_end != null && section.page_end !== section.page_start ? `–${section.page_end}` : ""}
                  </span>
                ) : null}
              </li>
            ))}
          </ol>
          {sections.length > INITIAL_VISIBLE ? (
            <button
              type="button"
              onClick={() => setShowAll((v) => !v)}
              className="mt-2 text-xs font-medium text-indigo-600 hover:underline dark:text-indigo-400"
            >
              {showAll ? "Show fewer" : `Show all ${sections.length} sections`}
            </button>
          ) : null}
        </>
      )}
    </section>
  );
}
