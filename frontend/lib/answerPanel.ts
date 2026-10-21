/**
 * State and presentation logic for the document Q&A panel, kept free of React so it can be
 * tested directly.
 *
 * Whether a citation may open a location is decided only by `citationLink()` (lib/answers.ts),
 * using the document versions the answer reports; the viewers then check the location again
 * against what they load. Everything here is plain text: nothing from the API is rendered as HTML.
 */

import type { DocxTarget } from "@/components/documents/DocxViewer";
import type { PdfTarget } from "@/components/documents/PdfViewer";
import type { AnswerRequest, AnswerResponse, Citation } from "@/lib/api/answers";
import { ApiError } from "@/lib/api/client";
import { citationLink, isNavigable, type CitationLink, type DocumentVersion } from "@/lib/answers";
import { codePointLength } from "@/lib/citations";

/** The server's default limit (RETRIEVAL_MAX_QUERY_CHARS); a lower server limit is still
 * reported by the API as QUERY_TOO_LONG. Counted in code points, as the server does. */
export const MAX_QUESTION_CHARS = 1000;

export type QuestionCheck = { ok: true; question: string } | { ok: false; reason: string | null };

/** A question ready to send, or why not (`reason` is null for an empty box: nothing to say). */
export function checkQuestion(text: string): QuestionCheck {
  const question = text.trim();
  if (!question) return { ok: false, reason: null };
  const length = codePointLength(question);
  if (length > MAX_QUESTION_CHARS) {
    return { ok: false, reason: `The question is too long (${length} of at most ${MAX_QUESTION_CHARS} characters).` };
  }
  return { ok: true, question };
}

// --- request state --------------------------------------------------------------------------------

export type PanelState =
  | { phase: "idle"; cancelled: boolean }
  | { phase: "loading"; requestId: number; question: string }
  | { phase: "answer"; requestId: number; question: string; response: AnswerResponse }
  | { phase: "error"; requestId: number; question: string; message: string };

export type PanelEvent =
  | { type: "submitted"; requestId: number; question: string }
  | { type: "answered"; requestId: number; response: AnswerResponse }
  | { type: "failed"; requestId: number; message: string }
  | { type: "cancelled"; requestId: number };

export const INITIAL_PANEL_STATE: PanelState = { phase: "idle", cancelled: false };

/** Only the request in flight may settle the panel: a late reply to a cancelled or replaced
 * request is ignored, and a second submission while one is in flight is ignored too. */
export function nextPanelState(state: PanelState, event: PanelEvent): PanelState {
  if (event.type === "submitted") {
    if (state.phase === "loading") return state;
    return { phase: "loading", requestId: event.requestId, question: event.question };
  }
  if (state.phase !== "loading" || state.requestId !== event.requestId) return state;
  if (event.type === "answered") {
    return { phase: "answer", requestId: state.requestId, question: state.question, response: event.response };
  }
  if (event.type === "failed") {
    return { phase: "error", requestId: state.requestId, question: state.question, message: event.message };
  }
  return { phase: "idle", cancelled: true };
}

export type Ask = (request: AnswerRequest, options: { signal: AbortSignal }) => Promise<AnswerResponse>;

/**
 * At most one answer request at a time. Cancelling aborts the HTTP request (the browser stops
 * waiting for it); the server may still finish generating that answer, but it is never shown.
 */
export class AnswerRequests {
  private current: { id: number; controller: AbortController } | null = null;
  private lastId = 0;

  constructor(
    private readonly ask: Ask,
    private readonly dispatch: (event: PanelEvent) => void,
  ) {}

  get busy(): boolean {
    return this.current !== null;
  }

  /** Send `text` about `documentId`; false (and nothing sent) if busy or the text is unusable. */
  submit(documentId: string, text: string): boolean {
    const check = checkQuestion(text);
    if (this.current || !check.ok) return false;
    const id = ++this.lastId;
    const controller = new AbortController();
    this.current = { id, controller };
    this.dispatch({ type: "submitted", requestId: id, question: check.question });
    this.ask({ question: check.question, document_ids: [documentId] }, { signal: controller.signal }).then(
      (response) => {
        if (this.settle(id)) this.dispatch({ type: "answered", requestId: id, response });
      },
      (error: unknown) => {
        if (!this.settle(id)) return; // cancelled or disposed: the outcome no longer matters
        this.dispatch(
          isAbort(error)
            ? { type: "cancelled", requestId: id }
            : { type: "failed", requestId: id, message: answerErrorMessage(error) },
        );
      },
    );
    return true;
  }

  cancel(): void {
    const current = this.current;
    if (!current) return;
    this.current = null;
    current.controller.abort();
    this.dispatch({ type: "cancelled", requestId: current.id });
  }

  /** Abort without reporting anything (the panel is going away). */
  dispose(): void {
    const current = this.current;
    this.current = null;
    current?.controller.abort();
  }

  private settle(id: number): boolean {
    if (this.current?.id !== id) return false;
    this.current = null;
    return true;
  }
}

export function isAbort(error: unknown): boolean {
  return typeof error === "object" && error !== null && (error as { name?: unknown }).name === "AbortError";
}

/** A short, plain-text explanation of a failed answer request. */
export function answerErrorMessage(error: unknown): string {
  if (!(error instanceof ApiError)) return "Something went wrong while answering. Please try again.";
  switch (error.code) {
    case "LLM_NOT_CONFIGURED":
      return "Question answering isn't configured on the server.";
    case "LLM_UNAVAILABLE":
      return "The language model is unavailable right now. Please try again shortly.";
    case "LLM_OUTPUT_INVALID":
      return "The model's reply was incomplete or malformed, so it wasn't shown. Please try again or rephrase the question.";
    case "DOCUMENT_NOT_INDEXED":
    case "DOCUMENT_BUSY":
      return "This document is being processed or needs reprocessing, so it can't be searched yet.";
    case "DOCUMENT_NOT_FOUND":
      return "This document no longer exists.";
    case "SERVICE_UNAVAILABLE":
      return "The service is temporarily unavailable. Please try again shortly.";
    case "TIMEOUT":
      return "The answer took too long and the request was stopped. Please try again.";
    default:
      // Server messages are plain text written for users (e.g. QUERY_TOO_LONG, NETWORK_ERROR).
      return error.message || "Something went wrong while answering. Please try again.";
  }
}

// --- answer presentation -------------------------------------------------------------------------

export interface GroundingSummary {
  label: string;
  tone: "good" | "partial" | "bad" | "neutral";
  description: string;
}

/** What exact quote verification establishes, and what it does not. */
export const SUPPORT_NOT_CHECKED =
  "Whether each quotation actually supports its statement is not checked automatically: read the quotes.";

/**
 * The label for an answer's grounding. Verification proves only that each quotation occurs word
 * for word in the document, so no label claims that a statement is supported, and "no answer
 * found" never claims the document lacks the information: only the excerpts searched were read.
 */
export function groundingSummary(response: Pick<AnswerResponse, "grounding" | "evidence">): GroundingSummary {
  switch (response.grounding) {
    case "grounded":
      return {
        label: "Quotes verified",
        tone: "good",
        description: `Every cited statement has a quotation found word for word in the document. ${SUPPORT_NOT_CHECKED}`,
      };
    case "partially_grounded":
      return {
        label: "Partly verified",
        tone: "partial",
        description: `Some statements have no verified quotation (see the notes); don't rely on those. ${SUPPORT_NOT_CHECKED}`,
      };
    case "ungrounded":
      return {
        label: "No verified quotations",
        tone: "bad",
        description:
          "No statement in the model's reply has a verified quotation it can be matched to, so the reply isn't presented as an answer.",
      };
    default: {
      const excerpts = response.evidence.length;
      return {
        label: "No answer found",
        tone: "neutral",
        description:
          (excerpts > 0
            ? `No answer was found in the ${excerpts} ${excerpts === 1 ? "excerpt" : "excerpts"} searched. `
            : "No passage was relevant enough to answer from, so the model wasn't asked. ") +
          "This doesn't prove the document lacks the information.",
      };
    }
  }
}

// The answer format's tags. The API removes them; this is a second line of defence (an older
// server, say), so a reply never shows raw "<answer>" markup. Other "<" text is left alone.
const PROTOCOL_TAG = /<\/?\s*(?:answer|citations|evidence)\b[^<>]{0,80}>/gi;

/** Model-written text (answer, draft, note) as it may be displayed: no protocol tags. */
export function displayText(text: string): string {
  return text.replace(PROTOCOL_TAG, "").trim();
}

export type AnswerSegment =
  | { kind: "text"; text: string }
  /**
   * A citation number in the answer. `index` is its citation's position in `citations`, or null
   * when it has none or when `conflict` is set: the number is used for more than one quotation,
   * so it cannot be linked to any one of them.
   */
  | { kind: "marker"; n: number; index: number | null; conflict: boolean };

// As the backend parses markers: "[3]", or several numbers in one bracket, "[3, 9]" / "[3,9]".
const MARKER = /\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]/g;

export function answerSegments(answer: string, citations: Citation[]): AnswerSegment[] {
  const segments: AnswerSegment[] = [];
  let position = 0;
  for (const match of answer.matchAll(MARKER)) {
    if (match.index > position) segments.push({ kind: "text", text: answer.slice(position, match.index) });
    for (const part of match[1].split(",")) {
      const n = Number(part.trim());
      const conflict = citations.some((c) => c.n === n && c.marker_conflict);
      const index = conflict ? -1 : citations.findIndex((c) => c.n === n);
      segments.push({ kind: "marker", n, index: index === -1 ? null : index, conflict });
    }
    position = match.index + match[0].length;
  }
  if (position < answer.length) segments.push({ kind: "text", text: answer.slice(position) });
  return segments;
}

export type NavigableLink = Exclude<CitationLink, { kind: "none" }>;

export interface CitationItem {
  index: number;
  citation: Citation;
  link: CitationLink;
  /** Set when the citation may be opened in the viewer now. */
  open: NavigableLink | null;
  /** Why it can't be opened (plain text), when it can't. */
  blocked: string | null;
  /** Where it is, for display: "Page 3", "Pages 3–4", or null. */
  pages: string | null;
  /** The section of the passage it was found in, if known. */
  section: string | null;
}

const VIEWER_UNAVAILABLE = "The document can't be shown right now (it may be processing), so this quote can't be opened.";

/**
 * Each citation of `response` with what clicking it may do in the viewer showing `viewed`.
 * Recompute whenever the viewed document changes: a document reprocessed after the answer no
 * longer matches the answer's version, so its citations stop being clickable.
 */
export function citationItems(
  response: Pick<AnswerResponse, "citations" | "documents" | "evidence">,
  viewed: DocumentVersion,
  viewerAvailable: boolean,
): CitationItem[] {
  const sections = new Map(response.evidence.map((e) => [e.evidence_id, e.section_path]));
  return response.citations.map((citation, index) => {
    const link = citationLink(citation, { viewed, answered: response.documents });
    const navigable = isNavigable(link) ? link : null;
    const verified = citation.status === "verified";
    const start = verified ? citation.page_start : null;
    const end = verified ? (citation.page_end ?? start) : null;
    return {
      index,
      citation,
      link,
      open: navigable && viewerAvailable ? navigable : null,
      blocked: link.kind === "none" ? link.reason : viewerAvailable ? null : VIEWER_UNAVAILABLE,
      pages: start == null ? null : end != null && end > start ? `Pages ${start}–${end}` : `Page ${start}`,
      section: verified && citation.evidence_id ? sections.get(citation.evidence_id) || null : null,
    };
  });
}

/** The citation the user chose, in the answer of request `requestId`. */
export interface CitationSelection {
  requestId: number;
  index: number;
}

export interface ViewerTargets {
  pdf?: PdfTarget;
  docx?: DocxTarget;
}

/**
 * The viewer target for the selected citation, or null if nothing (still) may be opened: the
 * selection belongs to an older answer, or the citation is no longer navigable. Returns new
 * objects, so choosing the same citation again scrolls to it again.
 */
export function selectedTarget(
  state: PanelState,
  items: CitationItem[],
  selection: CitationSelection | null,
): ViewerTargets | null {
  if (!selection || state.phase !== "answer" || state.requestId !== selection.requestId) return null;
  const open = items[selection.index]?.open;
  if (!open) return null;
  if (open.kind === "docx") return { docx: { ...open.target, ranges: open.target.ranges.map((r) => ({ ...r })) } };
  return { pdf: { ...open.target, highlights: open.target.highlights.map((r) => ({ ...r })) } };
}

/** What screen readers hear when the panel's state changes. */
export function panelAnnouncement(state: PanelState, items: CitationItem[]): string {
  switch (state.phase) {
    case "loading":
      return "Answering…";
    case "error":
      return "";
    case "idle":
      return state.cancelled ? "Request cancelled." : "";
    case "answer": {
      const verified = items.filter((i) => i.citation.status === "verified").length;
      return `Answer ready: ${groundingSummary(state.response).label}, ${verified} verified ${verified === 1 ? "quotation" : "quotations"}.`;
    }
  }
}
