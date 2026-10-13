import { AlertCircle, CheckCircle2, Loader2 } from "lucide-react";

import type { DocumentOut } from "@/lib/api/documents";
import { isInFlight, statusLabel } from "@/lib/documents";

export function StatusBadge({ document }: { document: Pick<DocumentOut, "status" | "stage" | "progress"> }) {
  const label = statusLabel(document);

  if (document.status === "ready") {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2 py-0.5 text-xs font-medium text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300">
        <CheckCircle2 aria-hidden className="h-3.5 w-3.5" />
        {label}
      </span>
    );
  }
  if (document.status === "failed") {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-red-50 px-2 py-0.5 text-xs font-medium text-red-700 dark:bg-red-950/60 dark:text-red-300">
        <AlertCircle aria-hidden className="h-3.5 w-3.5" />
        {label}
      </span>
    );
  }
  return (
    <span className="inline-flex min-w-0 flex-col gap-1">
      <span className="inline-flex items-center gap-1 text-xs font-medium text-indigo-700 dark:text-indigo-300">
        <Loader2 aria-hidden className="h-3.5 w-3.5 animate-spin" />
        {label}
        {document.status === "processing" ? <span className="tabular-nums text-zinc-500">· {document.progress}%</span> : null}
      </span>
      {isInFlight(document.status) ? (
        <span
          role="progressbar"
          aria-label={label}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={document.progress}
          className="block h-1 w-28 overflow-hidden rounded-full bg-indigo-100 dark:bg-indigo-950"
        >
          <span
            className="block h-full rounded-full bg-indigo-500 transition-[width] duration-500"
            style={{ width: `${Math.max(document.progress, 4)}%` }}
          />
        </span>
      ) : null}
    </span>
  );
}
