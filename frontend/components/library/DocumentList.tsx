"use client";

import { AlertTriangle, FileText, RotateCcw, Trash2 } from "lucide-react";
import Link from "next/link";

import { StatusBadge } from "@/components/library/StatusBadge";
import { buttonClass } from "@/components/ui/button";
import type { DocumentOut } from "@/lib/api/documents";
import {
  coverageSummary,
  describePages,
  formatBytes,
  formatDateTime,
  lowQualityNotice,
  unreadableNotice,
} from "@/lib/documents";

interface DocumentListProps {
  documents: DocumentOut[];
  onDelete: (document: DocumentOut) => void;
  onRetry: (document: DocumentOut) => void;
  retryingId: string | null;
}

export function DocumentList({ documents, onDelete, onRetry, retryingId }: DocumentListProps) {
  return (
    <ul className="divide-y divide-zinc-100 overflow-hidden rounded-xl border border-zinc-200 bg-white dark:divide-zinc-800 dark:border-zinc-800 dark:bg-zinc-900">
      {documents.map((document) => {
        const coverage = document.indexed ? coverageSummary(document) : describePages(document);
        const unreadable = document.indexed ? unreadableNotice(document) : null;
        const lowQuality = document.indexed ? lowQualityNotice(document) : null;
        const meta = [document.file_type.toUpperCase(), formatBytes(document.size_bytes), coverage, formatDateTime(document.created_at)]
          .filter(Boolean)
          .join(" · ");
        return (
          <li key={document.id} className="flex flex-col gap-3 px-4 py-3.5 sm:flex-row sm:items-center">
            <div className="flex min-w-0 flex-1 items-start gap-3">
              <FileText aria-hidden className="mt-0.5 h-5 w-5 shrink-0 text-zinc-400" />
              <div className="min-w-0">
                <Link
                  href={`/documents/${document.id}`}
                  className="block truncate font-medium hover:text-indigo-600 hover:underline dark:hover:text-indigo-400"
                >
                  {document.filename}
                </Link>
                <p className="mt-0.5 text-xs text-zinc-500">{meta}</p>
                {document.status === "failed" && document.error_message ? (
                  <p className="mt-1 text-xs text-red-700 dark:text-red-400">{document.error_message}</p>
                ) : null}
                {unreadable ? (
                  <p className="mt-1 flex items-center gap-1 text-xs text-amber-700 dark:text-amber-400">
                    <AlertTriangle aria-hidden className="h-3.5 w-3.5" />
                    {unreadable}
                  </p>
                ) : null}
                {lowQuality ? (
                  <p className="mt-1 flex items-center gap-1 text-xs text-amber-700 dark:text-amber-400">
                    <AlertTriangle aria-hidden className="h-3.5 w-3.5" />
                    {lowQuality}
                  </p>
                ) : null}
              </div>
            </div>
            <div className="flex items-center justify-between gap-2 pl-8 sm:justify-end sm:pl-0">
              <StatusBadge document={document} />
              <div className="flex items-center gap-1">
                {document.status === "failed" || document.needs_reprocessing ? (
                  <button
                    type="button"
                    className={buttonClass("ghost")}
                    onClick={() => onRetry(document)}
                    disabled={retryingId === document.id}
                    title={document.status === "failed" ? "Retry processing" : "Extract and index this document"}
                  >
                    <RotateCcw aria-hidden className="h-4 w-4" />
                    <span className="sr-only sm:not-sr-only">{document.status === "failed" ? "Retry" : "Reprocess"}</span>
                  </button>
                ) : null}
                <button
                  type="button"
                  className={buttonClass("ghost", "hover:text-red-600 dark:hover:text-red-400")}
                  onClick={() => onDelete(document)}
                  aria-label={`Delete ${document.filename}`}
                  title="Delete"
                >
                  <Trash2 aria-hidden className="h-4 w-4" />
                </button>
              </div>
            </div>
          </li>
        );
      })}
    </ul>
  );
}
