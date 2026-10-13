"use client";

import { useQuery } from "@tanstack/react-query";
import { ScanText } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Suspense } from "react";

import { getHealth } from "@/lib/api/health";

function HealthIndicator() {
  const { data, isError, isPending } = useQuery({
    queryKey: ["health"],
    queryFn: ({ signal }) => getHealth({ signal }),
    refetchInterval: 30_000,
    retry: false,
  });
  const tone = isPending
    ? "bg-zinc-300 dark:bg-zinc-600"
    : isError
      ? "bg-red-500"
      : data.status === "ok"
        ? "bg-emerald-500"
        : "bg-amber-500";
  const label = isPending
    ? "Checking backend"
    : isError
      ? "Backend unreachable"
      : data.status === "ok"
        ? "All systems operational"
        : "Backend degraded";
  return (
    <Link
      href="/status"
      title={label}
      className="inline-flex items-center gap-2 rounded-md px-2 py-1.5 text-xs text-zinc-600 hover:bg-zinc-100 dark:text-zinc-400 dark:hover:bg-zinc-800"
    >
      <span aria-hidden className={`h-2 w-2 rounded-full ${tone}`} />
      <span className="hidden sm:inline">{label}</span>
      <span className="sr-only sm:hidden">{label}</span>
    </Link>
  );
}

const NAV_ITEMS = [{ href: "/", label: "Library" }] as const;

const navClass = (active: boolean) =>
  `rounded-md px-2.5 py-1.5 text-sm font-medium ${
    active
      ? "bg-zinc-100 text-zinc-900 dark:bg-zinc-800 dark:text-zinc-100"
      : "text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100"
  }`;

function isActive(href: string, pathname: string): boolean {
  return href === "/" ? pathname === "/" || pathname.startsWith("/documents") : pathname.startsWith(href);
}

/** Reads the URL, so with Cache Components it must sit inside <Suspense> (see AppHeader). */
function ActiveNavLinks() {
  const pathname = usePathname();
  return NAV_ITEMS.map(({ href, label }) => (
    <Link key={href} href={href} aria-current={isActive(href, pathname) ? "page" : undefined} className={navClass(isActive(href, pathname))}>
      {label}
    </Link>
  ));
}

function PlainNavLinks() {
  return NAV_ITEMS.map(({ href, label }) => (
    <Link key={href} href={href} className={navClass(false)}>
      {label}
    </Link>
  ));
}

export function AppHeader() {
  return (
    <header className="sticky top-0 z-10 border-b border-zinc-200 bg-white/85 backdrop-blur dark:border-zinc-800 dark:bg-zinc-950/85">
      <div className="mx-auto flex h-14 w-full max-w-5xl items-center gap-4 px-4 sm:px-6">
        <Link href="/" className="flex items-center gap-2 font-semibold tracking-tight">
          <ScanText aria-hidden className="h-5 w-5 text-indigo-600 dark:text-indigo-400" />
          ClauseLens
        </Link>
        <nav aria-label="Main" className="flex items-center gap-1">
          <Suspense fallback={<PlainNavLinks />}>
            <ActiveNavLinks />
          </Suspense>
        </nav>
        <div className="ml-auto">
          <HealthIndicator />
        </div>
      </div>
    </header>
  );
}
