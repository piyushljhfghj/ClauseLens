import { Suspense } from "react";

import { DocumentDetailSkeleton } from "@/components/documents/DocumentDetailSkeleton";
import { DocumentRoute } from "@/components/documents/DocumentRoute";

// The document id is only known at request time. With Cache Components, reading it must happen
// inside <Suspense> so the skeleton can be prerendered as the static shell.
export default function DocumentPage() {
  return (
    <Suspense fallback={<DocumentDetailSkeleton />}>
      <DocumentRoute />
    </Suspense>
  );
}
