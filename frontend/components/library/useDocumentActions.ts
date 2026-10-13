"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { ApiError } from "@/lib/api/client";
import { deleteDocument, reprocessDocument, type DocumentList, type DocumentOut } from "@/lib/api/documents";

/** Delete (with a confirmation step) and retry, keeping the cached list in sync. */
export function useDocumentActions({ onDeleted }: { onDeleted?: (id: string) => void } = {}) {
  const queryClient = useQueryClient();
  const [pendingDelete, setPendingDelete] = useState<DocumentOut | null>(null);

  const remove = useMutation({
    mutationFn: (id: string) => deleteDocument(id),
    onSuccess: (_, id) => {
      queryClient.setQueryData<DocumentList>(["documents"], (current) =>
        current ? { items: current.items.filter((d) => d.id !== id), total: current.total - 1 } : current,
      );
      queryClient.removeQueries({ queryKey: ["document", id] });
      setPendingDelete(null);
      onDeleted?.(id);
    },
  });

  const retry = useMutation({
    mutationFn: (id: string) => reprocessDocument(id),
    onSuccess: (document) => {
      queryClient.setQueryData(["document", document.id], document);
      void queryClient.invalidateQueries({ queryKey: ["documents"] });
    },
  });

  return {
    pendingDelete,
    requestDelete: (document: DocumentOut) => {
      remove.reset();
      setPendingDelete(document);
    },
    cancelDelete: () => setPendingDelete(null),
    confirmDelete: () => pendingDelete && remove.mutate(pendingDelete.id),
    deleting: remove.isPending,
    deleteError: remove.error instanceof ApiError ? remove.error.message : remove.error ? String(remove.error) : null,
    retry: (document: DocumentOut) => retry.mutate(document.id),
    retryingId: retry.isPending ? (retry.variables ?? null) : null,
    retryError: retry.error instanceof ApiError ? retry.error.message : null,
  };
}
