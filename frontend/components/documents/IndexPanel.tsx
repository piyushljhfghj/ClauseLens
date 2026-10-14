"use client";

import { useQuery } from "@tanstack/react-query";
import { AlertTriangle } from "lucide-react";

import { getPages, type DocumentOut, type PageOut } from "@/lib/api/documents";
import { pageCoverageKind, type PageCoverageKind, unreadableNotice } from "@/lib/documents";

const PAGE_TONE: Record<PageCoverageKind, string> = {
  text: "bg-emerald-400 dark:bg-emerald-500",
  ocr: "bg-sky-400 dark:bg-sky-500",
  // Striped amber: indexed, but only partly recognised. Distinct from red "not read".
  low_quality:
    "bg-amber-400 bg-[repeating-linear-gradient(135deg,transparent_0_2px,rgba(255,255,255,0.55)_2px_4px)] dark:bg-amber-500",
  unreadable: "bg-red-500",
  blank: "bg-zinc-200 dark:bg-zinc-700",
};

function pageTitle(page: PageOut): string {
  const kind = pageCoverageKind(page);
  const detail = {
    text: "text layer",
    ocr: `OCR${page.ocr_confidence != null ? `, ${Math.round(page.ocr_confidence * 100)}% confidence` : ""}`,
    low_quality: `OCR, ${Math.round((page.ocr_confidence ?? 0) * 100)}% confidence — low quality, content may be largely unrecognised`,
    unreadable: {
      ocr_failed: "OCR failed — not read",
      ocr_no_text: "no text recognised (handwriting, diagram or poor scan) — not read",
      ocr_unavailable: "scanned, OCR unavailable — not read",
    }[page.note ?? ""] ?? "not read",
    blank: "blank",
  }[kind];
  return `Page ${page.page_number}: ${detail}`;
}

function Stat({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <dt className="text-xs text-zinc-500">{label}</dt>
      <dd className="mt-0.5 text-lg font-semibold tabular-nums">{value}</dd>
    </div>
  );
}

/** What was extracted and indexed, and — just as important — what was not. */
export function IndexPanel({ document }: { document: DocumentOut }) {
  const pages = useQuery({
    queryKey: ["document", document.id, "pages", document.processed_at],
    queryFn: ({ signal }) => getPages(document.id, signal),
    enabled: document.indexed && document.file_type === "pdf",
  });
  const counts: Record<PageCoverageKind, number> = { text: 0, ocr: 0, low_quality: 0, unreadable: 0, blank: 0 };
  const warnings = document.extraction_warnings ?? [];
  for (const page of pages.data ?? []) counts[pageCoverageKind(page)] += 1;
  const unreadable = unreadableNotice(document);

  return (
    <section aria-labelledby="index-heading" className="rounded-xl border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
      <h2 id="index-heading" className="text-sm font-semibold">
        Extraction &amp; index
      </h2>
      <dl className="mt-3 grid grid-cols-3 gap-3">
        <Stat label={document.file_type === "pdf" ? "Pages" : "Format"} value={document.page_count ?? "DOCX"} />
        <Stat label="Sections" value={document.section_count ?? "—"} />
        <Stat label="Passages" value={document.chunk_count ?? "—"} />
      </dl>

      {document.file_type === "pdf" ? (
        <div className="mt-4">
          <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-zinc-600 dark:text-zinc-400">
            {(
              [
                ["text", "Text layer", counts.text],
                ["ocr", "OCR", counts.ocr],
                ["low_quality", "Low-quality OCR", counts.low_quality],
                ["unreadable", "Not read", counts.unreadable],
                ["blank", "Blank", counts.blank],
              ] as const
            )
              .filter(([, , n]) => n > 0)
              .map(([kind, label, n]) => (
                <span key={kind} className="inline-flex items-center gap-1.5">
                  <span aria-hidden className={`h-2.5 w-2.5 rounded-sm ${PAGE_TONE[kind]}`} />
                  {label}: {n}
                </span>
              ))}
          </div>
          {pages.isPending ? (
            <div className="mt-2 h-6 animate-pulse rounded bg-zinc-100 dark:bg-zinc-800" />
          ) : pages.isError ? (
            <p className="mt-2 text-xs text-red-600">Couldn&apos;t load page details: {pages.error.message}</p>
          ) : (
            <ol aria-label="How each page was read" className="mt-2 flex flex-wrap gap-0.5">
              {pages.data.map((page) => (
                <li
                  key={page.page_number}
                  title={pageTitle(page)}
                  aria-label={pageTitle(page)}
                  className={`h-4 w-2 rounded-[2px] ${PAGE_TONE[pageCoverageKind(page)]}`}
                />
              ))}
            </ol>
          )}
        </div>
      ) : null}

      {unreadable || warnings.length > 0 ? (
        <ul className="mt-4 space-y-1.5">
          {/* Backend warnings name the affected pages (incl. low-quality OCR); the summary notice
              replaces only the generic "could not be read" one. */}
          {[unreadable, ...warnings.filter((w) => !w.includes("could not be read"))]
            .filter(Boolean)
            .map((warning) => (
              <li key={warning} className="flex gap-2 text-xs text-amber-800 dark:text-amber-300">
                <AlertTriangle aria-hidden className="mt-px h-3.5 w-3.5 shrink-0" />
                {warning}
              </li>
            ))}
        </ul>
      ) : null}
    </section>
  );
}
