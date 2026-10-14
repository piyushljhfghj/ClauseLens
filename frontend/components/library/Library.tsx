"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, FolderOpen, RefreshCw, RotateCcw } from "lucide-react";
import { useCallback } from "react";

import { DocumentList } from "@/components/library/DocumentList";
import { UploadDropzone } from "@/components/library/UploadDropzone";
import { useDocumentActions } from "@/components/library/useDocumentActions";
import { buttonClass } from "@/components/ui/button";
import { ConfirmDialog } from "@/components/ui/ConfirmDialog";
import { listDocuments, type DocumentList as DocumentListData, type DocumentOut } from "@/lib/api/documents";
import { isInFlight } from "@/lib/documents";

const POLL_WHILE_PROCESSING_MS = 1_500;

export function Library() {
  const queryClient = useQueryClient();
  const documents = useQuery({
    queryKey: ["documents"],
    queryFn: ({ signal }) => listDocuments(signal),
    // Poll only while something is being processed; stop once everything has settled.
    refetchInterval: (query) =>
      query.state.data?.items.some((d) => isInFlight(d.status)) ? POLL_WHILE_PROCESSING_MS : false,
  });
  const actions = useDocumentActions();

  const onUploaded = useCallback(
    (document: DocumentOut) => {
      queryClient.setQueryData<DocumentListData>(["documents"], (current) =>
        current && !current.items.some((d) => d.id === document.id)
          ? { items: [document, ...current.items], total: current.total + 1 }
          : current,
      );
      void queryClient.invalidateQueries({ queryKey: ["documents"] });
    },
    [queryClient],
  );

  return (
    <main className="mx-auto w-full max-w-5xl flex-1 px-4 py-8 sm:px-6 sm:py-10">
      <div className="flex items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Document library</h1>
          <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
            Upload contracts to analyse. Files are checked and processed in the background.
          </p>
        </div>
        {documents.data ? (
          <span className="shrink-0 text-sm text-zinc-500">
            {documents.data.total} {documents.data.total === 1 ? "document" : "documents"}
          </span>
        ) : null}
      </div>

      <div className="mt-6">
        <UploadDropzone onUploaded={onUploaded} />
      </div>

      <section aria-label="Documents" className="mt-8">
        {documents.data && documents.data.items.some((d) => d.needs_reprocessing) ? (
          <div className="mb-3 flex flex-col gap-3 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900 sm:flex-row sm:items-center dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-200">
            <AlertTriangle aria-hidden className="h-5 w-5 shrink-0" />
            <p className="flex-1">
              {documents.data.items.filter((d) => d.needs_reprocessing).length} document(s) were stored before text
              extraction and indexing were available. They can&apos;t be searched until they are reprocessed.
            </p>
            <button
              type="button"
              className={buttonClass("secondary")}
              onClick={() => documents.data.items.filter((d) => d.needs_reprocessing).forEach(actions.retry)}
            >
              <RotateCcw aria-hidden className="h-4 w-4" />
              Reprocess all
            </button>
          </div>
        ) : null}

        {actions.retryError ? (
          <p role="alert" className="mb-3 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700 dark:bg-red-950/50 dark:text-red-300">
            {actions.retryError}
          </p>
        ) : null}

        {documents.isPending ? (
          <div aria-busy="true" aria-label="Loading documents" className="space-y-2">
            {[0, 1, 2].map((i) => (
              <div key={i} className="h-16 animate-pulse rounded-xl bg-zinc-200/70 dark:bg-zinc-800/70" />
            ))}
          </div>
        ) : documents.isError ? (
          <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-6 text-center dark:border-red-900 dark:bg-red-950/40">
            <p className="text-sm font-medium text-red-800 dark:text-red-300">Couldn&apos;t load your documents</p>
            <p className="mt-1 text-sm text-red-700 dark:text-red-400">{documents.error.message}</p>
            <button type="button" className={buttonClass("secondary", "mt-4")} onClick={() => void documents.refetch()}>
              <RefreshCw aria-hidden className="h-4 w-4" />
              Try again
            </button>
          </div>
        ) : documents.data.items.length === 0 ? (
          <div className="rounded-xl border border-zinc-200 bg-white px-4 py-12 text-center dark:border-zinc-800 dark:bg-zinc-900">
            <FolderOpen aria-hidden className="mx-auto h-8 w-8 text-zinc-300 dark:text-zinc-600" />
            <p className="mt-3 text-sm font-medium">No documents yet</p>
            <p className="mt-1 text-sm text-zinc-500">Upload a PDF or Word contract to get started.</p>
          </div>
        ) : (
          <DocumentList
            documents={documents.data.items}
            onDelete={actions.requestDelete}
            onRetry={actions.retry}
            retryingId={actions.retryingId}
          />
        )}
      </section>

      <ConfirmDialog
        open={actions.pendingDelete !== null}
        title="Delete document?"
        description={
          <>
            <span className="font-medium text-zinc-900 dark:text-zinc-100">{actions.pendingDelete?.filename}</span> and
            its stored file will be permanently deleted.
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
