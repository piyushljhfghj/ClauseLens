/** Shared button styles (plain class strings; no component library needed yet). */

const base =
  "inline-flex items-center justify-center gap-1.5 rounded-md text-sm font-medium transition-colors " +
  "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-500 " +
  "disabled:pointer-events-none disabled:opacity-50";

const variants = {
  primary: "bg-zinc-900 px-3.5 py-2 text-white hover:bg-zinc-700 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-300",
  secondary:
    "border border-zinc-200 bg-white px-3 py-1.5 hover:bg-zinc-50 dark:border-zinc-700 dark:bg-zinc-900 dark:hover:bg-zinc-800",
  danger: "bg-red-600 px-3.5 py-2 text-white hover:bg-red-700",
  ghost: "px-2 py-1.5 text-zinc-600 hover:bg-zinc-100 hover:text-zinc-900 dark:text-zinc-400 dark:hover:bg-zinc-800 dark:hover:text-zinc-100",
} as const;

export function buttonClass(variant: keyof typeof variants = "secondary", extra = ""): string {
  return `${base} ${variants[variant]} ${extra}`.trim();
}
