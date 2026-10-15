"use client";

import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, Info } from "lucide-react";
import type { PDFDocumentProxy, RenderTask } from "pdfjs-dist";
import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";

import { getPages, type PageOut } from "@/lib/api/documents";
import {
  checkHighlights,
  INITIAL_PAGE_STATE,
  needsRelease,
  nextPageState,
  overlayPosition,
  pageOverlay,
  parsePreviewHighlights,
  renderScale,
  renderedSizeMatches,
  type HighlightCheck,
  type HighlightRect,
} from "@/lib/pdfHighlights";

/**
 * What to show:
 * - `citation`: the `highlights` of a citation the backend verified, and its page;
 * - `link`: a page and/or preview rectangles from the page URL, to check the viewer before
 *   citations can be opened from answers. Shown as a preview, never as a verified quote.
 */
export type PdfTarget =
  | { kind: "citation"; highlights: HighlightRect[]; page: number | null }
  | { kind: "link"; page: number | null; highlights: string | null };

type Loaded =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; pdf: PDFDocumentProxy };

/** Pages within this distance of the visible area are rendered; others are released. */
const RENDER_MARGIN = "150% 0px";
/** Width changes are re-rendered only once they settle (window resizing, phone rotation). */
const RESIZE_SETTLE_MS = 200;

/**
 * A PDF rendered with PDF.js, one canvas per page, rendered only near the visible area. Page
 * placeholders are sized from the stored page sizes, so the layout (and any highlight) is in its
 * final place before the pages are drawn. Verified citation rectangles are drawn on top.
 *
 * Keyed by the processing version: after reprocessing, the viewer starts afresh, so page-size
 * notices or rendering state from the previous version never carry over.
 */
export function PdfViewer(props: {
  documentId: string;
  processedAt: string | null;
  fileUrl: string;
  target?: PdfTarget;
}) {
  return <PdfViewerForVersion key={`${props.documentId}:${props.processedAt ?? ""}`} {...props} />;
}

function PdfViewerForVersion({
  documentId,
  processedAt,
  fileUrl,
  target,
}: {
  documentId: string;
  processedAt: string | null;
  fileUrl: string;
  target?: PdfTarget;
}) {
  const pages = useQuery({
    queryKey: ["document", documentId, "pages", processedAt],
    queryFn: ({ signal }) => getPages(documentId, signal),
  });
  const loaded = usePdfDocument(fileUrl);
  const [mismatched, setMismatched] = useState<ReadonlySet<number>>(new Set());
  const reportMismatch = useCallback(
    (pageNumber: number) =>
      setMismatched((current) => (current.has(pageNumber) ? current : new Set(current).add(pageNumber))),
    [],
  );
  // Elements in state (callback refs), not ref objects: observers must attach when the element
  // exists, including when React re-attaches refs (it does so in development).
  const [scrollArea, setScrollArea] = useState<HTMLDivElement | null>(null);
  const firstHighlight = useRef<HTMLElement | null>(null);
  const targetPage = useRef<HTMLElement | null>(null);

  const pageList = useMemo(() => pages.data ?? [], [pages.data]);
  const view = useMemo(() => resolveTarget(target, pageList), [target, pageList]);

  useEffect(() => {
    // Bring the cited passage (or page) into view once the placeholders exist. Instant, not
    // smooth: smooth scrolling did not move nested scroll areas in a mobile viewport.
    const element = firstHighlight.current ?? targetPage.current;
    if (!element) return;
    element.scrollIntoView({ block: firstHighlight.current ? "center" : "start" });
    if (firstHighlight.current) firstHighlight.current.focus({ preventScroll: true });
  }, [view, pageList]);

  if (pages.isPending) return <ViewerSkeleton />;
  if (pages.isError) {
    return <ViewerError title="Couldn't load the document's pages" message={pages.error.message} />;
  }
  if (loaded.status === "error") {
    return <ViewerError title="This PDF couldn't be displayed" message={`${loaded.message} You can still download the original.`} />;
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <TargetNotice view={view} mismatched={mismatched} />
      <div
        ref={setScrollArea}
        aria-label="Document pages"
        role="region"
        // Focusable so keyboard users can scroll it (arrow keys, Page Up/Down, Space).
        tabIndex={0}
        className="max-h-[75vh] min-h-[50vh] flex-1 overflow-y-auto rounded-xl border border-zinc-200 bg-zinc-100 p-2 outline-none focus-visible:ring-2 focus-visible:ring-zinc-400 sm:p-4 dark:border-zinc-800 dark:bg-zinc-950 dark:focus-visible:ring-zinc-500"
      >
        <div className="mx-auto flex max-w-3xl flex-col gap-3">
          {pageList.map((page) => {
            const rects = view.check?.ok ? view.check.byPage.get(page.page_number) ?? [] : [];
            const isFirst = view.check?.ok && page.page_number === view.check.firstPage;
            return (
              <PdfPage
                key={page.page_number}
                page={page}
                pdf={loaded.status === "ready" ? loaded.pdf : null}
                root={scrollArea}
                rects={mismatched.has(page.page_number) ? [] : rects}
                highlightRef={isFirst ? firstHighlight : undefined}
                pageRef={view.page === page.page_number && !isFirst ? targetPage : undefined}
                onSizeMismatch={reportMismatch}
              />
            );
          })}
        </div>
      </div>
    </div>
  );
}

interface View {
  kind: "citation" | "link" | null;
  page: number | null;
  check: HighlightCheck | null;
}

function resolveTarget(target: PdfTarget | undefined, pages: PageOut[]): View {
  if (!target || pages.length === 0) return { kind: null, page: null, check: null };
  const exists = (n: number | null) => (n !== null && pages.some((p) => p.page_number === n) ? n : null);
  if (target.kind === "citation") {
    const check = checkHighlights(target.highlights, pages);
    return { kind: "citation", page: check.ok ? check.firstPage : exists(target.page), check };
  }
  const rects = target.highlights === null ? null : parsePreviewHighlights(target.highlights, pages);
  const check =
    target.highlights === null
      ? null
      : rects
        ? checkHighlights(rects, pages)
        : ({ ok: false, reason: "The link's highlight is malformed or refers to a missing page." } as const);
  return { kind: "link", page: check?.ok ? check.firstPage : exists(target.page), check };
}

function TargetNotice({ view, mismatched }: { view: View; mismatched: ReadonlySet<number> }) {
  if (view.kind === null) return null;
  const affected = view.check?.ok ? [...view.check.byPage.keys()].filter((p) => mismatched.has(p)) : [];
  if (affected.length > 0) {
    return (
      <Notice tone="warning">
        Page {affected.join(", ")} renders at a different size than when the document was processed, so the
        highlight isn&apos;t shown there.
      </Notice>
    );
  }
  if (view.check && !view.check.ok) {
    return (
      <Notice tone="warning">
        {view.check.reason}
        {view.page !== null ? ` Showing page ${view.page}.` : ""}
      </Notice>
    );
  }
  if (view.kind === "link") {
    return (
      <Notice tone="info">
        {view.check ? "Preview highlight from the link" : `Page ${view.page ?? "?"} from the link`}. This is a preview, not a
        verified citation.
      </Notice>
    );
  }
  return (
    <p role="status" className="sr-only">
      Cited passage highlighted on page {view.page}.
    </p>
  );
}

function Notice({ tone, children }: { tone: "info" | "warning"; children: React.ReactNode }) {
  const warning = tone === "warning";
  const Icon = warning ? AlertTriangle : Info;
  return (
    <p
      role="status"
      className={
        warning
          ? "flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-200"
          : "flex items-start gap-2 rounded-lg border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm text-zinc-600 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-400"
      }
    >
      <Icon aria-hidden className="mt-0.5 h-4 w-4 shrink-0" />
      <span>{children}</span>
    </p>
  );
}

function PdfPage({
  page,
  pdf,
  root,
  rects,
  highlightRef,
  pageRef,
  onSizeMismatch,
}: {
  page: PageOut;
  pdf: PDFDocumentProxy | null;
  root: HTMLDivElement | null;
  rects: HighlightRect[];
  highlightRef?: React.RefObject<HTMLElement | null>;
  pageRef?: React.RefObject<HTMLElement | null>;
  onSizeMismatch: (pageNumber: number) => void;
}) {
  const [element, setElement] = useState<HTMLElement | null>(null);
  const canvas = useRef<HTMLCanvasElement | null>(null);
  // Set once this page has requested its PDF.js page object (so it may hold resources).
  const touched = useRef(false);
  const [near, setNear] = useState(false);
  const [width, setWidth] = useState(0);
  const [renderWidth, setRenderWidth] = useState(0);
  const [state, dispatch] = useReducer(nextPageState, INITIAL_PAGE_STATE);

  // Attached synchronously during the commit (before the viewer's scroll effect runs), so a
  // target page is in `pageRef` in time. Changes only when this page becomes or stops being the
  // target; React then detaches the old callback (clearing that ref) and attaches the new one.
  const sectionRef = useCallback(
    (node: HTMLElement | null) => {
      setElement(node);
      if (pageRef) pageRef.current = node;
    },
    [pageRef],
  );

  useEffect(() => {
    if (!element || !root) return;
    const intersection = new IntersectionObserver(([entry]) => setNear(entry.isIntersecting), {
      root,
      rootMargin: RENDER_MARGIN,
    });
    const resize = new ResizeObserver(([entry]) => setWidth(Math.round(entry.contentRect.width)));
    intersection.observe(element);
    resize.observe(element);
    return () => {
      intersection.disconnect();
      resize.disconnect();
    };
  }, [element, root]);

  // Render at the first known width at once; later width changes only once they settle.
  useEffect(() => {
    if (width === renderWidth) return;
    const timer = setTimeout(() => setRenderWidth(width), renderWidth === 0 ? 0 : RESIZE_SETTLE_MS);
    return () => clearTimeout(timer);
  }, [width, renderWidth]);

  // Render near the visible area: into an offscreen canvas first, so the current bitmap stays
  // on screen until its replacement is ready (no blank flash when re-rendering at a new width).
  useEffect(() => {
    const visible = canvas.current;
    if (!pdf || !near || renderWidth === 0 || !visible) return;
    let task: RenderTask | null = null;
    let cancelled = false;
    (async () => {
      let offscreen: HTMLCanvasElement | null = null;
      try {
        touched.current = true;
        const pdfPage = await pdf.getPage(page.page_number);
        if (cancelled) return;
        const natural = pdfPage.getViewport({ scale: 1 }); // the page's own rotation applied
        if (!renderedSizeMatches(page, natural.width, natural.height)) onSizeMismatch(page.page_number);
        const viewport = pdfPage.getViewport({ scale: renderScale(renderWidth, natural.width, window.devicePixelRatio) });
        offscreen = document.createElement("canvas");
        offscreen.width = Math.floor(viewport.width);
        offscreen.height = Math.floor(viewport.height);
        task = pdfPage.render({ canvas: offscreen, viewport });
        await task.promise;
        if (cancelled) return;
        visible.width = offscreen.width;
        visible.height = offscreen.height;
        visible.getContext("2d")?.drawImage(offscreen, 0, 0);
        dispatch("drawn");
      } catch (error) {
        if (!cancelled && !isCancelled(error)) dispatch("failed");
      } finally {
        // Every exit (drawn, failed, cancelled, early return): free the temporary bitmap now
        // rather than waiting for garbage collection. The render has settled by this point.
        if (offscreen) {
          offscreen.width = 0;
          offscreen.height = 0;
        }
      }
    })();
    return () => {
      cancelled = true;
      task?.cancel();
    };
  }, [pdf, near, renderWidth, page, onSizeMismatch]);

  // Release a rendered page that leaves the render area: its bitmap, and PDF.js's parsed
  // resources for it. Pages never rendered are skipped (no worker requests when the PDF opens).
  // The canvas is cleared together with the state change, and only if the page has not come
  // back meanwhile, so a returning page keeps its bitmap or shows the placeholder, never a
  // blank page. PDF.js defers the clean-up while a render is finishing and cancels it if the
  // page is rendered again, so this cannot disturb an active or upcoming render.
  useEffect(() => {
    if (!needsRelease(near, pdf !== null, touched.current) || !pdf) return;
    let cancelled = false;
    pdf
      .getPage(page.page_number)
      .then((pdfPage) => {
        if (cancelled) return;
        const visible = canvas.current;
        if (visible) {
          visible.width = 0;
          visible.height = 0;
        }
        dispatch("released");
        pdfPage.cleanup();
        touched.current = false;
      })
      .catch(() => {
        // The document was destroyed meanwhile (viewer closed): nothing left to release.
      });
    return () => {
      cancelled = true;
    };
  }, [near, pdf, page]);

  const overlay = pageOverlay(state);
  return (
    <section
      ref={sectionRef}
      aria-label={`Page ${page.page_number}`}
      className="relative w-full overflow-hidden bg-white shadow-sm ring-1 ring-zinc-200 dark:ring-zinc-800"
      style={{ aspectRatio: `${page.width} / ${page.height}` }}
    >
      <canvas ref={canvas} aria-hidden className="absolute inset-0 h-full w-full" />
      {overlay === "placeholder" ? (
        <div className="absolute inset-0 flex items-center justify-center text-xs text-zinc-400">
          Page {page.page_number}
        </div>
      ) : overlay === "error" ? (
        <div role="status" className="absolute inset-0 flex items-center justify-center bg-white/85 p-4 text-center text-xs text-zinc-600">
          Page {page.page_number} couldn&apos;t be rendered. You can still download the original.
        </div>
      ) : null}
      {rects.map((rect, i) => (
        <mark
          key={i}
          ref={i === 0 ? highlightRef : undefined}
          tabIndex={i === 0 && highlightRef ? -1 : undefined}
          aria-label={i === 0 ? "Cited passage" : undefined}
          className="pointer-events-none absolute rounded-[2px] bg-amber-300/45 outline-none ring-1 ring-amber-500/70 mix-blend-multiply focus-visible:ring-2"
          style={overlayPosition(rect)}
        />
      ))}
    </section>
  );
}

function isCancelled(error: unknown): boolean {
  return typeof error === "object" && error !== null && (error as { name?: string }).name === "RenderingCancelledException";
}

/** Fetch the PDF once and open it with PDF.js (loaded only in the browser); destroyed when the
 * file changes or the viewer unmounts. */
function usePdfDocument(fileUrl: string): Loaded {
  const [loaded, setLoaded] = useState<{ url: string; state: Loaded }>({ url: fileUrl, state: { status: "loading" } });

  useEffect(() => {
    const controller = new AbortController();
    // Destroying the loading task also destroys the document and frees the worker's memory.
    let destroy: (() => Promise<void>) | null = null;
    let cancelled = false;
    (async () => {
      try {
        const pdfjs = await import("pdfjs-dist");
        pdfjs.GlobalWorkerOptions.workerSrc = new URL("pdfjs-dist/build/pdf.worker.min.mjs", import.meta.url).toString();
        const response = await fetch(fileUrl, { signal: controller.signal });
        if (!response.ok) throw new Error(`The file couldn't be downloaded (HTTP ${response.status}).`);
        const data = new Uint8Array(await response.arrayBuffer());
        if (cancelled) return;
        const task = pdfjs.getDocument({ data });
        destroy = () => task.destroy();
        const pdf = await task.promise;
        if (cancelled) return;
        setLoaded({ url: fileUrl, state: { status: "ready", pdf } });
      } catch (error) {
        if (cancelled || controller.signal.aborted) return;
        const message = error instanceof Error && error.name === "InvalidPDFException"
          ? "The file isn't a valid PDF."
          : error instanceof Error
            ? error.message
            : "Unknown error.";
        setLoaded({ url: fileUrl, state: { status: "error", message } });
      }
    })();
    return () => {
      cancelled = true;
      controller.abort();
      if (destroy) void destroy();
    };
  }, [fileUrl]);

  return loaded.url === fileUrl ? loaded.state : { status: "loading" };
}

function ViewerSkeleton() {
  return (
    <div className="flex min-h-[50vh] flex-col gap-3 rounded-xl border border-zinc-200 bg-zinc-100 p-4 dark:border-zinc-800 dark:bg-zinc-950" aria-busy>
      <div className="mx-auto aspect-[595/842] w-full max-w-3xl animate-pulse rounded bg-white dark:bg-zinc-900" />
    </div>
  );
}

function ViewerError({ title, message }: { title: string; message: string }) {
  return (
    <div role="alert" className="rounded-xl border border-zinc-200 bg-white px-4 py-12 text-center text-sm dark:border-zinc-800 dark:bg-zinc-900">
      <p className="font-medium">{title}</p>
      <p className="mt-1 text-zinc-500">{message}</p>
    </div>
  );
}
