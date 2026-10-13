"use client";

import { AlertCircle, CheckCircle2, FileUp, Loader2, X } from "lucide-react";
import Link from "next/link";
import { useCallback, useRef, useState } from "react";

import { ApiError } from "@/lib/api/client";
import { uploadDocument, type DocumentOut } from "@/lib/api/documents";
import { ACCEPT_ATTRIBUTE, MAX_UPLOAD_MB, formatBytes, validateUploadFile } from "@/lib/documents";

type UploadItem = {
  id: number;
  name: string;
  size: number;
  state: "pending" | "uploading" | "done" | "error";
  progress: number;
  error?: string;
  existingDocumentId?: string;
};

const DONE_ITEM_TTL_MS = 3_000;

export function UploadDropzone({ onUploaded }: { onUploaded: (document: DocumentOut) => void }) {
  const [items, setItems] = useState<UploadItem[]>([]);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const queue = useRef<{ id: number; file: File }[]>([]);
  const running = useRef(false);
  const nextId = useRef(1);

  const update = useCallback((id: number, patch: Partial<UploadItem>) => {
    setItems((current) => current.map((item) => (item.id === id ? { ...item, ...patch } : item)));
  }, []);

  const remove = useCallback((id: number) => {
    setItems((current) => current.filter((item) => item.id !== id));
  }, []);

  // Uploads run one at a time so several large files don't compete for bandwidth.
  const pump = useCallback(async () => {
    if (running.current) return;
    running.current = true;
    while (queue.current.length > 0) {
      const { id, file } = queue.current.shift()!;
      update(id, { state: "uploading" });
      try {
        const document = await uploadDocument(file, (fraction) => update(id, { progress: fraction })).promise;
        update(id, { state: "done", progress: 1 });
        onUploaded(document);
        setTimeout(() => remove(id), DONE_ITEM_TTL_MS);
      } catch (err) {
        const error = err instanceof ApiError ? err : new ApiError("UNKNOWN", String(err));
        const existing = error.code === "DUPLICATE_DOCUMENT" ? error.details.document_id : undefined;
        update(id, {
          state: "error",
          error: error.message,
          existingDocumentId: typeof existing === "string" ? existing : undefined,
        });
      }
    }
    running.current = false;
  }, [onUploaded, remove, update]);

  const addFiles = useCallback(
    (files: FileList | File[]) => {
      const added: UploadItem[] = [];
      for (const file of Array.from(files)) {
        const id = nextId.current++;
        const problem = validateUploadFile(file);
        added.push({
          id,
          name: file.name,
          size: file.size,
          state: problem ? "error" : "pending",
          progress: 0,
          error: problem ?? undefined,
        });
        if (!problem) queue.current.push({ id, file });
      }
      setItems((current) => [...current, ...added]);
      void pump();
    },
    [pump],
  );

  return (
    <div>
      <label
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          if (event.dataTransfer.files.length) addFiles(event.dataTransfer.files);
        }}
        className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed px-6 py-10 text-center transition-colors focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-indigo-500 ${
          dragging
            ? "border-indigo-500 bg-indigo-50 dark:bg-indigo-950/40"
            : "border-zinc-300 bg-white hover:border-zinc-400 dark:border-zinc-700 dark:bg-zinc-900 dark:hover:border-zinc-500"
        }`}
      >
        <FileUp aria-hidden className="h-8 w-8 text-zinc-400" />
        <span className="text-sm font-medium">
          Drop contracts here, or <span className="text-indigo-600 underline dark:text-indigo-400">browse</span>
        </span>
        <span className="text-xs text-zinc-500">PDF or Word (.docx), up to {MAX_UPLOAD_MB} MB each</span>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={ACCEPT_ATTRIBUTE}
          className="sr-only"
          onChange={(event) => {
            if (event.target.files?.length) addFiles(event.target.files);
            event.target.value = ""; // allow choosing the same file again
          }}
        />
      </label>

      {items.length > 0 ? (
        <ul aria-live="polite" className="mt-3 space-y-2">
          {items.map((item) => (
            <li
              key={item.id}
              className="flex items-start gap-3 rounded-lg border border-zinc-200 bg-white px-3 py-2.5 text-sm dark:border-zinc-800 dark:bg-zinc-900"
            >
              <span className="mt-0.5">
                {item.state === "error" ? (
                  <AlertCircle aria-hidden className="h-4 w-4 text-red-500" />
                ) : item.state === "done" ? (
                  <CheckCircle2 aria-hidden className="h-4 w-4 text-emerald-500" />
                ) : (
                  <Loader2 aria-hidden className="h-4 w-4 animate-spin text-indigo-500" />
                )}
              </span>
              <div className="min-w-0 flex-1">
                <div className="flex items-baseline justify-between gap-3">
                  <span className="truncate font-medium">{item.name}</span>
                  <span className="shrink-0 text-xs text-zinc-500">{formatBytes(item.size)}</span>
                </div>
                {item.state === "uploading" || item.state === "pending" ? (
                  <div className="mt-1.5 flex items-center gap-2">
                    <div
                      role="progressbar"
                      aria-label={`Uploading ${item.name}`}
                      aria-valuemin={0}
                      aria-valuemax={100}
                      aria-valuenow={Math.round(item.progress * 100)}
                      className="h-1 flex-1 overflow-hidden rounded-full bg-zinc-100 dark:bg-zinc-800"
                    >
                      <div
                        className="h-full rounded-full bg-indigo-500 transition-[width]"
                        style={{ width: `${Math.round(item.progress * 100)}%` }}
                      />
                    </div>
                    <span className="w-24 text-right text-xs tabular-nums text-zinc-500">
                      {item.state === "pending"
                        ? "Waiting…"
                        : item.progress >= 1
                          ? "Validating…"
                          : `Uploading ${Math.round(item.progress * 100)}%`}
                    </span>
                  </div>
                ) : null}
                {item.state === "done" ? (
                  <p className="mt-0.5 text-xs text-emerald-700 dark:text-emerald-400">Uploaded. Processing has started.</p>
                ) : null}
                {item.state === "error" ? (
                  <p role="alert" className="mt-0.5 text-xs text-red-700 dark:text-red-400">
                    {item.error}{" "}
                    {item.existingDocumentId ? (
                      <Link href={`/documents/${item.existingDocumentId}`} className="font-medium underline">
                        Open it
                      </Link>
                    ) : null}
                  </p>
                ) : null}
              </div>
              {item.state === "error" ? (
                <button
                  type="button"
                  aria-label={`Dismiss ${item.name}`}
                  onClick={() => remove(item.id)}
                  className="rounded p-0.5 text-zinc-400 hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-800"
                >
                  <X aria-hidden className="h-4 w-4" />
                </button>
              ) : null}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
