export function DocumentDetailSkeleton() {
  return (
    <main aria-busy="true" aria-label="Loading document" className="mx-auto w-full max-w-5xl flex-1 px-4 py-8 sm:px-6">
      <div className="h-8 w-24 animate-pulse rounded-md bg-zinc-200/70 dark:bg-zinc-800/70" />
      <div className="mt-4 h-24 animate-pulse rounded-xl bg-zinc-200/70 dark:bg-zinc-800/70" />
      <div className="mt-6 h-[50vh] animate-pulse rounded-xl bg-zinc-200/50 dark:bg-zinc-800/50" />
    </main>
  );
}
