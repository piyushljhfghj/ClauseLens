import { describe, expect, it } from "vitest";

import { askDocuments, type AnswerRequest, type AnswerResponse, type Citation } from "@/lib/api/answers";
import { ApiError } from "@/lib/api/client";
import type { DocumentVersion } from "@/lib/answers";
import {
  AnswerRequests,
  answerErrorMessage,
  answerSegments,
  checkQuestion,
  citationItems,
  displayText,
  groundingSummary,
  INITIAL_PANEL_STATE,
  MAX_QUESTION_CHARS,
  nextPanelState,
  panelAnnouncement,
  selectedTarget,
  type Ask,
  type PanelEvent,
  type PanelState,
} from "@/lib/answerPanel";

const PDF_ID = "11111111-1111-1111-1111-111111111111";
const DOCX_ID = "22222222-2222-2222-2222-222222222222";
const V1 = "2026-10-01T10:00:00.123456Z";
const V2 = "2026-10-02T09:00:00.000001Z";

const pdfDoc: DocumentVersion = { id: PDF_ID, file_type: "pdf", processed_at: V1 };
const docxDoc: DocumentVersion = { id: DOCX_ID, file_type: "docx", processed_at: V1 };
const rect = { page: 2, page_width: 612, page_height: 792, x0: 72, y0: 100, x1: 300, y1: 114 };
const range = { block: 4, kind: "paragraph", block_char_start: 480, block_char_end: 560, start: 20, end: 46 };

function citation(overrides: Partial<Citation> = {}): Citation {
  return {
    n: 1,
    status: "verified",
    proposed_quote: "The Supplier shall deliver",
    cited_evidence_id: "E1",
    evidence_id: "E1",
    document_id: PDF_ID,
    document_filename: "contract.pdf",
    quote: "The Supplier shall deliver",
    char_start: 500,
    char_end: 526,
    page_start: 2,
    page_end: 2,
    highlights: [rect],
    block_ranges: [],
    occurrence_count: 1,
    location_ambiguous: false,
    referenced: true,
    marker_conflict: false,
    reason: null,
    ...overrides,
  };
}

function response(overrides: Partial<AnswerResponse> = {}): AnswerResponse {
  return {
    question: "What must the supplier do?",
    answer: "The supplier must deliver [1].",
    grounding: "grounded",
    citations: [citation()],
    evidence: [
      {
        evidence_id: "E1",
        document_id: PDF_ID,
        document_filename: "contract.pdf",
        chunk_id: 1,
        section_path: "4 Delivery",
        char_start: 450,
        char_end: 700,
        page_start: 2,
        page_end: 2,
        text: "…",
        retrieval_rank: 1,
        quality: { level: "text" } as AnswerResponse["evidence"][number]["quality"],
      },
    ],
    excluded: [],
    coverage: [],
    documents: [pdfDoc],
    notes: [],
    ...overrides,
  };
}

const run = (...events: PanelEvent[]) => events.reduce(nextPanelState, INITIAL_PANEL_STATE);

// --- question input -------------------------------------------------------------------------------

describe("checkQuestion", () => {
  it("rejects an empty or blank question without a message", () => {
    expect(checkQuestion("")).toEqual({ ok: false, reason: null });
    expect(checkQuestion("  \n\t ")).toEqual({ ok: false, reason: null });
  });

  it("trims the question", () => {
    expect(checkQuestion("  What is the cap?\n")).toEqual({ ok: true, question: "What is the cap?" });
  });

  it("counts code points like the server and explains an over-long question", () => {
    expect(checkQuestion("a".repeat(MAX_QUESTION_CHARS)).ok).toBe(true);
    // 999 letters + one emoji is 1000 code points (but 1001 UTF-16 units): still allowed.
    expect(checkQuestion("a".repeat(MAX_QUESTION_CHARS - 1) + "😀").ok).toBe(true);
    const tooLong = checkQuestion("a".repeat(MAX_QUESTION_CHARS + 1));
    expect(tooLong.ok).toBe(false);
    if (!tooLong.ok) expect(tooLong.reason).toContain("too long");
  });
});

// --- request state --------------------------------------------------------------------------------

describe("nextPanelState", () => {
  it("goes from loading to an answer or an error", () => {
    const loading = run({ type: "submitted", requestId: 1, question: "q" });
    expect(loading).toEqual({ phase: "loading", requestId: 1, question: "q" });
    expect(nextPanelState(loading, { type: "answered", requestId: 1, response: response() }).phase).toBe("answer");
    expect(nextPanelState(loading, { type: "failed", requestId: 1, message: "m" })).toMatchObject({ phase: "error", message: "m" });
  });

  it("ignores a second submission while one is in flight", () => {
    const loading = run({ type: "submitted", requestId: 1, question: "q" });
    expect(nextPanelState(loading, { type: "submitted", requestId: 2, question: "other" })).toBe(loading);
  });

  it("treats a cancelled request as cancelled, not failed, and ignores its late reply", () => {
    const cancelled = run({ type: "submitted", requestId: 1, question: "q" }, { type: "cancelled", requestId: 1 });
    expect(cancelled).toEqual({ phase: "idle", cancelled: true });
    expect(nextPanelState(cancelled, { type: "answered", requestId: 1, response: response() })).toBe(cancelled);
    expect(nextPanelState(cancelled, { type: "failed", requestId: 1, message: "m" })).toBe(cancelled);
  });

  it("ignores replies to an older request", () => {
    const loading = run({ type: "submitted", requestId: 2, question: "q" });
    expect(nextPanelState(loading, { type: "answered", requestId: 1, response: response() })).toBe(loading);
  });

  it("allows asking again after an answer, an error or a cancellation", () => {
    for (const settled of [
      { type: "answered", requestId: 1, response: response() },
      { type: "failed", requestId: 1, message: "m" },
      { type: "cancelled", requestId: 1 },
    ] as PanelEvent[]) {
      const state = run({ type: "submitted", requestId: 1, question: "q" }, settled, { type: "submitted", requestId: 2, question: "q2" });
      expect(state).toEqual({ phase: "loading", requestId: 2, question: "q2" });
    }
  });
});

/** An `Ask` whose replies the test controls. */
function controlledAsk() {
  const calls: { request: AnswerRequest; signal: AbortSignal; resolve: (r: AnswerResponse) => void; reject: (e: unknown) => void }[] = [];
  const ask: Ask = (request, { signal }) =>
    new Promise((resolve, reject) => {
      calls.push({ request, signal, resolve, reject });
    });
  return { ask, calls };
}

function recorder() {
  const events: PanelEvent[] = [];
  let state: PanelState = INITIAL_PANEL_STATE;
  return {
    events,
    dispatch: (event: PanelEvent) => {
      events.push(event);
      state = nextPanelState(state, event);
    },
    get state() {
      return state;
    },
  };
}

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

describe("AnswerRequests", () => {
  it("asks about the viewed document only, with the trimmed question", async () => {
    const { ask, calls } = controlledAsk();
    const panel = recorder();
    const requests = new AnswerRequests(ask, panel.dispatch);
    expect(requests.submit(PDF_ID, "  What is the cap? ")).toBe(true);
    expect(calls[0].request).toEqual({ question: "What is the cap?", document_ids: [PDF_ID] });
    expect(requests.busy).toBe(true);
    calls[0].resolve(response());
    await settle();
    expect(panel.state.phase).toBe("answer");
    expect(requests.busy).toBe(false);
  });

  it("sends nothing for an empty question", () => {
    const { ask, calls } = controlledAsk();
    const panel = recorder();
    expect(new AnswerRequests(ask, panel.dispatch).submit(PDF_ID, "   ")).toBe(false);
    expect(calls).toHaveLength(0);
    expect(panel.events).toHaveLength(0);
  });

  it("refuses a duplicate submission while a request is in flight", () => {
    const { ask, calls } = controlledAsk();
    const requests = new AnswerRequests(ask, recorder().dispatch);
    requests.submit(PDF_ID, "q");
    expect(requests.submit(PDF_ID, "q")).toBe(false);
    expect(calls).toHaveLength(1);
  });

  it("shows API failures as a plain-language error", async () => {
    const { ask, calls } = controlledAsk();
    const panel = recorder();
    new AnswerRequests(ask, panel.dispatch).submit(PDF_ID, "q");
    calls[0].reject(new ApiError("LLM_UNAVAILABLE", "upstream 503", 503));
    await settle();
    expect(panel.state).toMatchObject({ phase: "error", message: expect.stringContaining("unavailable right now") });
  });

  it("cancels: aborts the request, reports a cancellation (not an error) and ignores the late reply", async () => {
    const { ask, calls } = controlledAsk();
    const panel = recorder();
    const requests = new AnswerRequests(ask, panel.dispatch);
    requests.submit(PDF_ID, "q");
    requests.cancel();
    expect(calls[0].signal.aborted).toBe(true);
    expect(panel.state).toEqual({ phase: "idle", cancelled: true });
    expect(requests.busy).toBe(false);
    calls[0].resolve(response()); // the server finished anyway
    await settle();
    expect(panel.state).toEqual({ phase: "idle", cancelled: true });
    expect(panel.events.map((e) => e.type)).toEqual(["submitted", "cancelled"]);
  });

  it("can ask again right after cancelling, and the old request cannot overwrite the new one", async () => {
    const { ask, calls } = controlledAsk();
    const panel = recorder();
    const requests = new AnswerRequests(ask, panel.dispatch);
    requests.submit(PDF_ID, "first");
    requests.cancel();
    expect(requests.submit(PDF_ID, "second")).toBe(true);
    calls[0].resolve(response({ answer: "old" }));
    calls[1].resolve(response({ answer: "new" }));
    await settle();
    expect(panel.state).toMatchObject({ phase: "answer", question: "second", response: { answer: "new" } });
  });

  it("treats an abort it did not start as a cancellation, never as a failure", async () => {
    const { ask, calls } = controlledAsk();
    const panel = recorder();
    new AnswerRequests(ask, panel.dispatch).submit(PDF_ID, "q");
    calls[0].reject(new DOMException("The operation was aborted.", "AbortError"));
    await settle();
    expect(panel.state).toEqual({ phase: "idle", cancelled: true });
  });

  it("dispose aborts without reporting anything", async () => {
    const { ask, calls } = controlledAsk();
    const panel = recorder();
    const requests = new AnswerRequests(ask, panel.dispatch);
    requests.submit(PDF_ID, "q");
    requests.dispose();
    expect(calls[0].signal.aborted).toBe(true);
    calls[0].reject(new DOMException("aborted", "AbortError"));
    await settle();
    expect(panel.events.map((e) => e.type)).toEqual(["submitted"]);
  });

  it("cancels a real askDocuments request through the HTTP client", async () => {
    let fetchSignal: AbortSignal | undefined;
    const fetchImpl: typeof fetch = (_input, init) =>
      new Promise((_resolve, reject) => {
        fetchSignal = init?.signal ?? undefined;
        fetchSignal?.addEventListener("abort", () => reject(fetchSignal?.reason));
      });
    const panel = recorder();
    const requests = new AnswerRequests(
      (request, options) => askDocuments(request, { ...options, baseUrl: "http://api.test", fetchImpl }),
      panel.dispatch,
    );
    requests.submit(PDF_ID, "q");
    await settle();
    requests.cancel();
    await settle();
    expect(fetchSignal?.aborted).toBe(true);
    expect(panel.state).toEqual({ phase: "idle", cancelled: true });
  });
});

describe("answerErrorMessage", () => {
  it.each([
    ["LLM_NOT_CONFIGURED", "isn't configured"],
    ["LLM_OUTPUT_INVALID", "incomplete or malformed"],
    ["DOCUMENT_NOT_INDEXED", "being processed"],
    ["DOCUMENT_BUSY", "being processed"],
    ["DOCUMENT_NOT_FOUND", "no longer exists"],
    ["TIMEOUT", "took too long"],
  ])("explains %s", (code, text) => {
    expect(answerErrorMessage(new ApiError(code, "raw"))).toContain(text);
  });

  it("passes other API messages through as text", () => {
    const message = "The query is too long (maximum 1000 characters).";
    expect(answerErrorMessage(new ApiError("QUERY_TOO_LONG", message, 422))).toBe(message);
  });

  it("has a fallback for unexpected errors", () => {
    expect(answerErrorMessage(new TypeError("x is undefined"))).toContain("Something went wrong");
  });
});

// --- answer presentation -------------------------------------------------------------------------

describe("groundingSummary", () => {
  const summary = (grounding: AnswerResponse["grounding"], excerpts = 4) =>
    groundingSummary({ grounding, evidence: response().evidence.concat(Array(Math.max(0, excerpts - 1)).fill(response().evidence[0])).slice(0, excerpts) });

  it("labels each grounding", () => {
    expect(summary("grounded")).toMatchObject({ label: "Quotes verified", tone: "good" });
    expect(summary("partially_grounded")).toMatchObject({ label: "Partly verified", tone: "partial" });
    expect(summary("ungrounded")).toMatchObject({ label: "No verified quotations", tone: "bad" });
    expect(summary("not_found")).toMatchObject({ label: "No answer found", tone: "neutral" });
  });

  it("never claims that quotations support their statements", () => {
    for (const grounding of ["grounded", "partially_grounded", "ungrounded", "not_found"] as const) {
      const { label, description } = summary(grounding);
      expect(`${label} ${description}`).not.toMatch(/\bgrounded\b|backed by|supported by/i);
    }
    expect(summary("grounded").description).toContain("not checked automatically");
    expect(summary("partially_grounded").description).toContain("not checked automatically");
  });

  it("says 'no answer found' only about the excerpts searched, never about the whole document", () => {
    const found = summary("not_found", 4).description;
    expect(found).toContain("4 excerpts searched");
    expect(found).toContain("doesn't prove the document lacks");
    expect(found).not.toMatch(/document doesn't (appear to )?answer/i);
    expect(summary("not_found", 1).description).toContain("1 excerpt searched");
    expect(summary("not_found", 0).description).toContain("model wasn't asked");
  });
});

describe("displayText", () => {
  it("removes nested or stray answer-format tags (the observed '<answer>' leak)", () => {
    expect(displayText("<answer>I couldn't find sufficient support.</answer>")).toBe("I couldn't find sufficient support.");
    expect(displayText("<answer>\n<answer>Text [1].</answer>\n</answer>")).toBe("Text [1].");
    expect(displayText('A [1]. </citations><evidence id="E2">')).toBe("A [1].");
  });

  it("keeps ordinary angle brackets and never interprets markup", () => {
    expect(displayText("Notice < 5 days and <b>bold</b>")).toBe("Notice < 5 days and <b>bold</b>");
    expect(displayText("<answering machine>")).toBe("<answering machine>");
  });
});

describe("answerSegments", () => {
  it("splits markers from text and links them to their citations", () => {
    const citations = [citation({ n: 1 }), citation({ n: 2 })];
    expect(answerSegments("Pay in 30 days [2] or terminate [1][3].", citations)).toEqual([
      { kind: "text", text: "Pay in 30 days " },
      { kind: "marker", n: 2, index: 1, conflict: false },
      { kind: "text", text: " or terminate " },
      { kind: "marker", n: 1, index: 0, conflict: false },
      { kind: "marker", n: 3, index: null, conflict: false },
      { kind: "text", text: "." },
    ]);
  });

  it("keeps markup in the answer as literal text", () => {
    expect(answerSegments("<b>bold</b> <script>x</script>", [])).toEqual([
      { kind: "text", text: "<b>bold</b> <script>x</script>" },
    ]);
  });
});

describe("citationItems", () => {
  it("opens a verified PDF quote at its highlight, with page and section", () => {
    const [item] = citationItems(response(), pdfDoc, true);
    expect(item.open).toMatchObject({ kind: "pdf", page: 2 });
    expect(item).toMatchObject({ blocked: null, pages: "Page 2", section: "4 Delivery" });
  });

  it("opens a verified page-only PDF quote at its page, with an explanation", () => {
    const [item] = citationItems(response({ citations: [citation({ highlights: [], page_end: 3 })] }), pdfDoc, true);
    expect(item.open).toMatchObject({ kind: "page", page: 2 });
    expect(item.open?.kind === "page" && item.open.explanation).toContain("exact position");
    expect(item.pages).toBe("Pages 2–3");
  });

  it("opens a verified Word quote at its text ranges", () => {
    const answer = response({
      citations: [citation({ document_id: DOCX_ID, page_start: null, page_end: null, highlights: [], block_ranges: [range] })],
      documents: [docxDoc],
    });
    const [item] = citationItems(answer, docxDoc, true);
    expect(item.open).toMatchObject({ kind: "docx" });
    expect(item.pages).toBeNull();
  });

  it("never opens an unverified quote, and shows no location for it", () => {
    const [item] = citationItems(response({ citations: [citation({ status: "unverified", reason: "Not found." })] }), pdfDoc, true);
    expect(item.open).toBeNull();
    expect(item.blocked).toContain("could not be verified");
    expect(item).toMatchObject({ pages: null, section: null });
  });

  it("stops opening quotes once the document is reprocessed after the answer", () => {
    const answer = response();
    expect(citationItems(answer, pdfDoc, true)[0].open).not.toBeNull();
    const [item] = citationItems(answer, { ...pdfDoc, processed_at: V2 }, true);
    expect(item.open).toBeNull();
    expect(item.blocked).toContain("reprocessed");
  });

  it("does not open quotes when the answer has no version for the document", () => {
    const [item] = citationItems(response({ documents: [] }), pdfDoc, true);
    expect(item.open).toBeNull();
    expect(item.link).toMatchObject({ code: "version_unknown" });
  });

  it("does not open quotes of another document", () => {
    const [item] = citationItems(response(), { ...docxDoc, id: "33333333-3333-3333-3333-333333333333" }, true);
    expect(item.link).toMatchObject({ code: "other_document" });
  });

  it("does not open quotes with an invalid location", () => {
    const [item] = citationItems(response({ citations: [citation({ highlights: [{ ...rect, page: 9 }] })] }), pdfDoc, true);
    expect(item.open).toBeNull();
    expect(item.link).toMatchObject({ code: "invalid_location" });
  });

  it("does not open quotes while the viewer can't show the document", () => {
    const [item] = citationItems(response(), pdfDoc, false);
    expect(item.open).toBeNull();
    expect(item.blocked).toContain("can't be shown right now");
  });

  it("does not open an ambiguous quote, and explains the ambiguity", () => {
    const [item] = citationItems(
      response({ citations: [citation({ location_ambiguous: true, occurrence_count: 2 })] }),
      pdfDoc,
      true,
    );
    expect(item.open).toBeNull();
    expect(item.link).toMatchObject({ code: "ambiguous_location" });
    expect(item.blocked).toContain("2 times");
    expect(item.pages).toBe("Page 2"); // the verified quote and its page are still shown
  });
});

describe("selectedTarget", () => {
  const answered = run(
    { type: "submitted", requestId: 7, question: "q" },
    { type: "answered", requestId: 7, response: response() },
  );

  it("targets the chosen PDF quote, with new objects on every call", () => {
    const items = citationItems(response(), pdfDoc, true);
    const first = selectedTarget(answered, items, { requestId: 7, index: 0 });
    const again = selectedTarget(answered, items, { requestId: 7, index: 0 });
    expect(first).toEqual({ pdf: { kind: "citation", highlights: [rect], page: 2 } });
    expect(again).not.toBe(first);
    expect(again?.pdf).not.toBe(first?.pdf);
  });

  it("targets a page-only quote as a page with no rectangles", () => {
    const answer = response({ citations: [citation({ highlights: [] })] });
    const items = citationItems(answer, pdfDoc, true);
    expect(selectedTarget(answered, items, { requestId: 7, index: 0 })).toEqual({
      pdf: { kind: "citation", highlights: [], page: 2 },
    });
  });

  it("targets a Word quote's ranges", () => {
    const answer = response({
      citations: [citation({ document_id: DOCX_ID, page_start: null, page_end: null, highlights: [], block_ranges: [range] })],
      documents: [docxDoc],
    });
    const items = citationItems(answer, docxDoc, true);
    expect(selectedTarget(answered, items, { requestId: 7, index: 0 })).toEqual({ docx: { kind: "citation", ranges: [range] } });
  });

  it("drops the target once the document is reprocessed", () => {
    const stale = citationItems(response(), { ...pdfDoc, processed_at: V2 }, true);
    expect(selectedTarget(answered, stale, { requestId: 7, index: 0 })).toBeNull();
  });

  it("ignores a selection from an earlier answer or while a new question is loading", () => {
    const items = citationItems(response(), pdfDoc, true);
    expect(selectedTarget(answered, items, { requestId: 6, index: 0 })).toBeNull();
    const loading = nextPanelState({ phase: "idle", cancelled: false }, { type: "submitted", requestId: 8, question: "q" });
    expect(selectedTarget(loading, items, { requestId: 7, index: 0 })).toBeNull();
  });

  it("ignores an unverified or missing citation", () => {
    const items = citationItems(response({ citations: [citation({ status: "unverified" })] }), pdfDoc, true);
    expect(selectedTarget(answered, items, { requestId: 7, index: 0 })).toBeNull();
    expect(selectedTarget(answered, items, { requestId: 7, index: 5 })).toBeNull();
  });
});

describe("panelAnnouncement", () => {
  it("announces loading, cancellation and the answer", () => {
    expect(panelAnnouncement({ phase: "loading", requestId: 1, question: "q" }, [])).toBe("Answering…");
    expect(panelAnnouncement({ phase: "idle", cancelled: true }, [])).toBe("Request cancelled.");
    const answer = response({ grounding: "partially_grounded", citations: [citation(), citation({ n: 2, status: "unverified" })] });
    const state: PanelState = { phase: "answer", requestId: 1, question: "q", response: answer };
    expect(panelAnnouncement(state, citationItems(answer, pdfDoc, true))).toBe("Answer ready: Partly verified, 1 verified quotation.");
  });

  it("leaves errors to the alert and says nothing when idle", () => {
    expect(panelAnnouncement({ phase: "error", requestId: 1, question: "q", message: "m" }, [])).toBe("");
    expect(panelAnnouncement(INITIAL_PANEL_STATE, [])).toBe("");
  });
});


describe("Pass A review: citation markers", () => {
  it("reads combined markers as separate numbers, each linked on its own (A06)", () => {
    const citations = [citation({ n: 3 }), citation({ n: 9 })];
    for (const text of ["Relates to communication [3, 9].", "Relates to communication [3,9]."]) {
      expect(answerSegments(text, citations)).toEqual([
        { kind: "text", text: "Relates to communication " },
        { kind: "marker", n: 3, index: 0, conflict: false },
        { kind: "marker", n: 9, index: 1, conflict: false },
        { kind: "text", text: "." },
      ]);
    }
  });

  it("never links a number used for more than one quotation (A04)", () => {
    const citations = [
      citation({ n: 4, marker_conflict: true }),
      citation({ n: 4, quote: "Long term goals", marker_conflict: true }),
    ];
    const markers = answerSegments("Short term [4]. Long term [4].", citations).filter((s) => s.kind === "marker");
    expect(markers).toEqual([
      { kind: "marker", n: 4, index: null, conflict: true },
      { kind: "marker", n: 4, index: null, conflict: true },
    ]);
  });

  it("leaves bracketed text that is not a marker as plain text", () => {
    expect(answerSegments("See [E3] and [3-5] and [a, b].", [citation()])).toEqual([
      { kind: "text", text: "See [E3] and [3-5] and [a, b]." },
    ]);
  });

  it("describes an unsupported reply without claiming no quotation was found", () => {
    const { description } = groundingSummary({ grounding: "ungrounded", evidence: [] });
    expect(description).toContain("verified quotation it can be matched to");
    expect(description).not.toMatch(/were found in the document/);
  });
});
