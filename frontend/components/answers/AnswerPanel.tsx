"use client";

import { useRef, useState } from "react";
import { CircleCheck, CircleX, Locate, LoaderCircle, MessageSquareText, TriangleAlert, X } from "lucide-react";

import type { DocumentQa } from "@/components/answers/useDocumentQa";
import { buttonClass } from "@/components/ui/button";
import type { AnswerResponse } from "@/lib/api/answers";
import {
  answerSegments,
  checkQuestion,
  displayText,
  groundingSummary,
  panelAnnouncement,
  SUPPORT_NOT_CHECKED,
  type CitationItem,
  type GroundingSummary,
} from "@/lib/answerPanel";

const TONE: Record<GroundingSummary["tone"], string> = {
  good: "border-emerald-200 bg-emerald-50 text-emerald-800 dark:border-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-300",
  partial: "border-amber-200 bg-amber-50 text-amber-900 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-200",
  bad: "border-red-200 bg-red-50 text-red-800 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300",
  neutral: "border-zinc-200 bg-zinc-50 text-zinc-700 dark:border-zinc-700 dark:bg-zinc-800/60 dark:text-zinc-300",
};

/**
 * Ask a question about the document shown and read the answer with its quotations. A verified
 * quote with a usable location opens in the viewer; everything else says why it can't.
 * All text from the API is rendered as text, never as HTML.
 */
export function AnswerPanel({ qa, available }: { qa: DocumentQa; available: boolean }) {
  const [text, setText] = useState("");
  const check = checkQuestion(text);
  const loading = qa.state.phase === "loading";
  const canSubmit = available && check.ok && !loading;

  function submit() {
    if (canSubmit) qa.submit(text);
  }

  return (
    <section
      aria-labelledby="qa-heading"
      className="flex flex-col gap-3 rounded-xl border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900"
    >
      <h2 id="qa-heading" className="flex items-center gap-2 text-sm font-semibold">
        <MessageSquareText aria-hidden className="h-4 w-4 text-zinc-400" />
        Ask this document
      </h2>

      <form
        className="flex flex-col gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          submit();
        }}
      >
        <label htmlFor="qa-question" className="sr-only">
          Question about this document
        </label>
        <textarea
          id="qa-question"
          rows={3}
          value={text}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
              event.preventDefault();
              submit();
            }
          }}
          disabled={!available}
          aria-describedby="qa-hint"
          aria-invalid={!check.ok && check.reason !== null}
          placeholder="e.g. How can either party terminate the agreement?"
          className="w-full resize-y rounded-md border border-zinc-200 bg-white px-3 py-2 text-sm outline-none placeholder:text-zinc-400 focus-visible:ring-2 focus-visible:ring-indigo-500 disabled:opacity-60 dark:border-zinc-700 dark:bg-zinc-950"
        />
        <p id="qa-hint" className={`text-xs ${!check.ok && check.reason ? "text-red-600 dark:text-red-400" : "text-zinc-500"}`}>
          {!available
            ? "Questions can be asked once the document is processed and indexed."
            : !check.ok && check.reason
              ? check.reason
              : "Enter to ask · Shift+Enter for a new line"}
        </p>
        <div className="flex gap-2">
          <button type="submit" className={buttonClass("primary", "flex-1")} disabled={!canSubmit}>
            {loading ? <LoaderCircle aria-hidden className="h-4 w-4 animate-spin" /> : null}
            {loading ? "Answering…" : "Ask"}
          </button>
          {loading ? (
            <button type="button" className={buttonClass("secondary")} onClick={qa.cancel}>
              <X aria-hidden className="h-4 w-4" />
              Cancel
            </button>
          ) : null}
        </div>
      </form>

      <p role="status" className="sr-only">
        {panelAnnouncement(qa.state, qa.items)}
      </p>

      <PanelBody qa={qa} />
    </section>
  );
}

function PanelBody({ qa }: { qa: DocumentQa }) {
  const { state } = qa;
  if (state.phase === "idle") {
    return (
      <p className="text-sm text-zinc-500">
        {state.cancelled
          ? "Request cancelled. The server may still finish generating that answer, but it won't be shown."
          : "Answers quote the document. Each quotation is checked word for word against its text; whether it supports its statement is for you to judge."}
      </p>
    );
  }
  if (state.phase === "loading") {
    return (
      <div className="flex flex-col gap-2" aria-hidden>
        <p className="text-sm text-zinc-500">Searching the document and checking quotes. This can take up to a minute.</p>
        <div className="h-3 w-5/6 animate-pulse rounded bg-zinc-100 dark:bg-zinc-800" />
        <div className="h-3 w-2/3 animate-pulse rounded bg-zinc-100 dark:bg-zinc-800" />
      </div>
    );
  }
  if (state.phase === "error") {
    return (
      <div role="alert" className="flex gap-2 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300">
        <CircleX aria-hidden className="mt-0.5 h-4 w-4 shrink-0" />
        <p>{state.message}</p>
      </div>
    );
  }
  return <Answer qa={qa} question={state.question} response={state.response} />;
}

function Answer({ qa, question, response }: { qa: DocumentQa; question: string; response: AnswerResponse }) {
  const summary = groundingSummary(response);
  const answer = displayText(response.answer);
  const draft = response.unsupported_draft ? displayText(response.unsupported_draft) : "";
  const cards = useRef(new Map<number, HTMLElement>());

  function openMarker(index: number | null) {
    if (index === null) return;
    if (qa.items[index]?.open) qa.select(index);
    else cards.current.get(index)?.focus(); // show why it can't be opened
  }

  return (
    <div className="flex flex-col gap-3">
      <p className="text-xs text-zinc-500">
        <span className="font-medium text-zinc-700 dark:text-zinc-300">Q:</span> {question}
      </p>
      <div className={`rounded-md border px-3 py-2 text-xs ${TONE[summary.tone]}`}>
        <p className="font-semibold">{summary.label}</p>
        <p className="mt-0.5">{summary.description}</p>
      </div>

      <p className="whitespace-pre-wrap break-words text-sm leading-relaxed">
        {answerSegments(answer, response.citations).map((segment, i) =>
          segment.kind === "text" ? (
            <span key={i}>{segment.text}</span>
          ) : segment.conflict ? (
            <span
              key={i}
              className="text-amber-700 dark:text-amber-400"
              title="This number is used for more than one quotation, so it can't be matched to one"
            >
              [{segment.n}]
              <span className="sr-only"> (number used for more than one quotation)</span>
            </span>
          ) : segment.index === null ? (
            <span key={i} className="text-zinc-400" title="This marker has no citation">
              [{segment.n}]
              <span className="sr-only"> (no citation)</span>
            </span>
          ) : (
            <button
              key={i}
              type="button"
              onClick={() => openMarker(segment.index)}
              aria-label={`Quote ${segment.n}`}
              className="mx-0.5 rounded px-0.5 align-baseline text-xs font-semibold text-indigo-600 hover:underline focus-visible:outline-2 focus-visible:outline-indigo-500 dark:text-indigo-400"
            >
              [{segment.n}]
            </button>
          ),
        )}
      </p>

      {draft ? (
        <details className="rounded-md border border-red-200 p-2 text-xs dark:border-red-900">
          <summary className="cursor-pointer font-medium text-red-700 dark:text-red-400">
            Show the model&apos;s unverified reply
          </summary>
          <p className="mt-2 text-zinc-500">None of its quotations were found in the document. Don&apos;t rely on it.</p>
          <p className="mt-1 whitespace-pre-wrap break-words text-zinc-700 dark:text-zinc-300">{draft}</p>
        </details>
      ) : null}

      {qa.items.length > 0 ? (
        <ol aria-label="Quotes" className="flex flex-col gap-2">
          {qa.items.map((item) => (
            <CitationCard
              key={item.index}
              item={item}
              selected={qa.selected === item.index}
              onOpen={() => qa.select(item.index)}
              cardRef={(node) => {
                if (node) cards.current.set(item.index, node);
                else cards.current.delete(item.index);
              }}
            />
          ))}
        </ol>
      ) : null}

      {response.notes.length > 0 ? (
        <details className="text-xs text-zinc-500">
          <summary className="cursor-pointer">Notes ({response.notes.length})</summary>
          <ul className="mt-1 list-disc space-y-1 pl-4">
            {response.notes.map((note, i) => (
              <li key={i}>{displayText(note)}</li>
            ))}
          </ul>
        </details>
      ) : null}
    </div>
  );
}

function CitationCard({
  item,
  selected,
  onOpen,
  cardRef,
}: {
  item: CitationItem;
  selected: boolean;
  onOpen: () => void;
  cardRef: (node: HTMLElement | null) => void;
}) {
  const { citation, open } = item;
  const verified = citation.status === "verified";
  const where = [item.pages, item.section].filter(Boolean).join(" · ");
  const ambiguous = item.link.kind === "none" && item.link.code === "ambiguous_location";
  // The backend's own explanation, where it adds something (why unverified, why no location).
  const serverDetail =
    item.link.kind === "none" && (item.link.code === "unverified" || item.link.code === "no_location")
      ? item.link.serverReason
      : null;
  return (
    <li
      ref={cardRef}
      tabIndex={-1}
      aria-label={`Quote ${citation.n}, ${verified ? "found in the document" : "not found in the document"}`}
      className={`rounded-md border p-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${
        selected ? "border-indigo-400 bg-indigo-50/60 dark:border-indigo-700 dark:bg-indigo-950/30" : "border-zinc-200 dark:border-zinc-800"
      }`}
    >
      <div className="flex items-center gap-2 text-xs">
        <span className="font-semibold">[{citation.n}]</span>
        {verified ? (
          <span
            className="inline-flex items-center gap-1 text-emerald-700 dark:text-emerald-400"
            title={`Found word for word in the document. ${SUPPORT_NOT_CHECKED}`}
          >
            <CircleCheck aria-hidden className="h-3.5 w-3.5" />
            Quote verified
          </span>
        ) : (
          <span className="inline-flex items-center gap-1 text-red-700 dark:text-red-400">
            <CircleX aria-hidden className="h-3.5 w-3.5" />
            Not found in document
          </span>
        )}
        {where ? <span className="truncate text-zinc-500">{where}</span> : null}
      </div>

      {verified ? (
        <blockquote className="mt-2 whitespace-pre-wrap break-words border-l-2 border-zinc-300 pl-2 text-zinc-800 dark:border-zinc-600 dark:text-zinc-200">
          {citation.quote}
        </blockquote>
      ) : (
        <>
          <p className="mt-2 text-xs text-zinc-500">The model&apos;s proposed quote, which was not found in the document:</p>
          <p className="mt-1 whitespace-pre-wrap break-words text-zinc-500 line-through decoration-zinc-300">
            {citation.proposed_quote}
          </p>
        </>
      )}

      {citation.marker_conflict ? (
        <p className="mt-2 flex gap-1.5 text-xs text-amber-800 dark:text-amber-300">
          <TriangleAlert aria-hidden className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          This quotation&apos;s number is also used for a different quotation, so it can&apos;t be matched to a
          statement and supports none of them.
        </p>
      ) : !citation.referenced ? (
        <p className="mt-2 flex gap-1.5 text-xs text-amber-800 dark:text-amber-300">
          <TriangleAlert aria-hidden className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          No marker in the answer refers to this quotation, so it supports none of its statements.
        </p>
      ) : null}
      {ambiguous ? (
        <p className="mt-2 flex gap-1.5 text-xs text-amber-800 dark:text-amber-300">
          <TriangleAlert aria-hidden className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          {item.blocked}
        </p>
      ) : null}
      {open?.kind === "page" ? <p className="mt-2 text-xs text-zinc-500">{open.explanation}</p> : null}

      {open ? (
        <button type="button" className={buttonClass("secondary", "mt-2 text-xs")} onClick={onOpen}>
          <Locate aria-hidden className="h-3.5 w-3.5" />
          {open.kind === "page" ? `Go to page ${open.page}` : "Show in document"}
        </button>
      ) : item.blocked && !ambiguous ? (
        <p className="mt-2 text-xs text-zinc-500">
          {item.blocked}
          {serverDetail ? ` ${serverDetail}` : ""}
        </p>
      ) : null}
    </li>
  );
}
