"use client";

import { useParams } from "next/navigation";

import { DocumentDetail } from "@/components/documents/DocumentDetail";

export function DocumentRoute() {
  const { id } = useParams<{ id: string }>();
  // Keyed by id so navigating between documents resets local state (e.g. an open dialog).
  return <DocumentDetail key={id} id={id} />;
}
