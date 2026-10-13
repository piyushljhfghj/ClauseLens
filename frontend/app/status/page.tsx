import type { Metadata } from "next";

import { BackendStatus } from "@/components/system/BackendStatus";

export const metadata: Metadata = { title: "System status · ClauseLens" };

export default function StatusPage() {
  return (
    <main className="mx-auto w-full max-w-3xl flex-1 px-4 py-8 sm:px-6 sm:py-10">
      <h1 className="text-2xl font-semibold tracking-tight">System status</h1>
      <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
        Live checks of the API, database, vector extension and migrations.
      </p>
      <div className="mt-6">
        <BackendStatus />
      </div>
    </main>
  );
}
