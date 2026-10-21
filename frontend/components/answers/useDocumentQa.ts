"use client";

import { useCallback, useEffect, useMemo, useReducer, useState } from "react";

import { askDocuments } from "@/lib/api/answers";
import type { DocumentOut } from "@/lib/api/documents";
import {
  AnswerRequests,
  citationItems,
  INITIAL_PANEL_STATE,
  nextPanelState,
  selectedTarget,
  type CitationItem,
  type CitationSelection,
  type PanelState,
  type ViewerTargets,
} from "@/lib/answerPanel";

export interface DocumentQa {
  state: PanelState;
  items: CitationItem[];
  /** Viewer target of the chosen citation, while it may still be opened. */
  target: ViewerTargets | null;
  selected: number | null;
  submit: (text: string) => boolean;
  cancel: () => void;
  select: (index: number) => void;
}

/**
 * Questions about one document (the one shown) and the citation chosen from the answer. Mount
 * once per document: the page is keyed by document id, so nothing carries over to another one.
 */
export function useDocumentQa(document: DocumentOut | undefined): DocumentQa {
  const [state, dispatch] = useReducer(nextPanelState, INITIAL_PANEL_STATE);
  const [requests] = useState(() => new AnswerRequests((request, options) => askDocuments(request, options), dispatch));
  const [selection, setSelection] = useState<CitationSelection | null>(null);

  // Leaving the page aborts a request still in flight.
  useEffect(() => () => requests.dispose(), [requests]);

  const id = document?.id;
  const fileType = document?.file_type;
  const processedAt = document?.processed_at ?? null;
  const viewerAvailable = document?.indexed ?? false;
  const items = useMemo(
    () =>
      state.phase === "answer" && id && fileType
        ? citationItems(state.response, { id, file_type: fileType, processed_at: processedAt }, viewerAvailable)
        : [],
    [state, id, fileType, processedAt, viewerAvailable],
  );
  const target = useMemo(() => selectedTarget(state, items, selection), [state, items, selection]);

  const submit = useCallback((text: string) => (id ? requests.submit(id, text) : false), [id, requests]);
  const cancel = useCallback(() => requests.cancel(), [requests]);
  const requestId = state.phase === "answer" ? state.requestId : null;
  const select = useCallback(
    (index: number) => {
      // A new object every time, so choosing the same citation again scrolls to it again.
      if (requestId !== null) setSelection({ requestId, index });
    },
    [requestId],
  );

  return {
    state,
    items,
    target,
    selected: target && selection ? selection.index : null,
    submit,
    cancel,
    select,
  };
}
