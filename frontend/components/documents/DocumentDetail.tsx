"use client";

import { useQuery } from "@tanstack/react-query";
import { AlertCircle, AlertTriangle, ArrowLeft, Download, FileText, RotateCcw, Trash2 } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";

import { DocumentDetailSkeleton } from "@/components/documents/DocumentDetailSkeleton";
import { StatusBadge } from "@/components/library/StatusBadge";
import { useDocumentActions } from "@/components/library/useDocumentActions";
import { buttonClass } from "@/components/ui/button";
import { ConfirmDialog } from "@/components/ui/ConfirmDialog";
import { ApiError } from "@/lib/api/client";
import { documentFileUrl, getDocument } from "@/lib/api/documents";
import { describePages, formatBytes, formatDateTime, isInFlight } from "@/lib/documents";

export function DocumentDetail({ id }: { id: string }) {
  const router = useRouter();
  const query = useQuery({
    queryKey: ["document", id],
    queryFn: ({ signal }) => getDocument(id, signal),
    refetchInterval: (q) => (q.state.data && isInFlight(q.state.data.status) ? 1_500 : false),
  });
  const actions = useDocumentActions({ onDeleted: () => router.push("/") });

  const backLink = (
    <Link href="/" className={buttonClass("ghost", "-ml-2 self-start")}>
      <ArrowLeft aria-hidden className="h-4 w-4" />
      Library
    </Link>
  );

  if (query.isPending) return <DocumentDetailSkeleton />;

  if (query.isError) {
    const notFound = query.error instanceof ApiError && (query.error.status === 404 || query.error.status === 422);
    return (
      <main className="mx-auto w-full max-w-5xl flex-1 px-4 py-8 sm:px-6">
        {backLink}
        <div className="mt-4 rounded-xl border border-zinc-200 bg-white px-4 py-12 text-center dark:border-zinc-800 dark:bg-zinc-900">
          <p className="font-medium">{notFound ? "Document not found" : "Couldn't load this document"}</p>
          <p className="mt-1 text-sm text-zinc-500">
            {notFound ? "It may have been deleted." : query.error.message}
          </p>
        </div>
      </main>
    );
  }

  const document = query.data;
  const pages = describePages(document);

  return (
    <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col px-4 py-8 sm:px-6">
      {backLink}

      <header className="mt-3 flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex min-w-0 items-start gap-3">
          <FileText aria-hidden className="mt-1 h-6 w-6 shrink-0 text-zinc-400" />
          <div className="min-w-0">
            <h1 className="break-words text-xl font-semibold tracking-tight">{document.filename}</h1>
            <p className="mt-1 text-sm text-zinc-500">
              {[document.file_type.toUpperCase(), formatBytes(document.size_bytes), pages, `Uploaded ${formatDateTime(document.created_at)}`]
                .filter(Boolean)
                .join(" · ")}
            </p>
            <div className="mt-2">
              <StatusBadge document={document} />
            </div>
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <a href={documentFileUrl(document.id, { download: true })} className={buttonClass("secondary")}>
            <Download aria-hidden className="h-4 w-4" />
            Download
          </a>
          <button type="button" className={buttonClass("secondary", "hover:text-red-600")} onClick={() => actions.requestDelete(document)}>
            <Trash2 aria-hidden className="h-4 w-4" />
            Delete
          </button>
        </div>
      </header>

      {document.status === "failed" ? (
        <div role="alert" className="mt-6 flex flex-col gap-3 rounded-xl border border-red-200 bg-red-50 p-4 sm:flex-row sm:items-start dark:border-red-900 dark:bg-red-950/40">
          <AlertCircle aria-hidden className="h-5 w-5 shrink-0 text-red-600" />
          <div className="flex-1">
            <p className="font-medium text-red-800 dark:text-red-300">This document couldn&apos;t be processed</p>
            <p className="mt-1 text-sm text-red-700 dark:text-red-400">{document.error_message}</p>
            {actions.retryError ? <p className="mt-1 text-sm text-red-700">{actions.retryError}</p> : null}
          </div>
          <button
            type="button"
            className={buttonClass("secondary")}
            onClick={() => actions.retry(document)}
            disabled={actions.retryingId === document.id}
          >
            <RotateCcw aria-hidden className="h-4 w-4" />
            Retry processing
          </button>
        </div>
      ) : null}

      {document.status === "ready" && document.pages_without_text ? (
        <div className="mt-6 flex gap-3 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-300">
          <AlertTriangle aria-hidden className="h-5 w-5 shrink-0" />
          <p>
            {document.pages_without_text} of {document.page_count} pages have no selectable text (they look scanned).
            Their contents can&apos;t be read from the text layer.
          </p>
        </div>
      ) : null}

      <section aria-label="Document" className="mt-6 flex flex-1 flex-col">
        {document.status !== "ready" ? (
          <div className="rounded-xl border border-dashed border-zinc-300 px-4 py-12 text-center text-sm text-zinc-500 dark:border-zinc-700">
            {isInFlight(document.status)
              ? "The document will be available here once processing finishes."
              : "Processing failed, so the document isn't available for analysis. You can still download the original."}
          </div>
        ) : document.file_type === "pdf" ? (
          <iframe
            title={`Preview of ${document.filename}`}
            src={documentFileUrl(document.id)}
            className="min-h-[70vh] w-full flex-1 rounded-xl border border-zinc-200 bg-white dark:border-zinc-800"
          />
        ) : (
          <div className="rounded-xl border border-zinc-200 bg-white px-4 py-12 text-center text-sm text-zinc-600 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-400">
            In-app preview isn&apos;t available for Word documents yet. Use <span className="font-medium">Download</span> to open the original.
          </div>
        )}
      </section>

      <ConfirmDialog
        open={actions.pendingDelete !== null}
        title="Delete document?"
        description={
          <>
            <span className="font-medium text-zinc-900 dark:text-zinc-100">{document.filename}</span> and its stored file
            will be permanently deleted.
          </>
        }
        confirmLabel="Delete"
        busy={actions.deleting}
        error={actions.deleteError}
        onConfirm={actions.confirmDelete}
        onCancel={actions.cancelDelete}
      />
    </main>
  );
}
