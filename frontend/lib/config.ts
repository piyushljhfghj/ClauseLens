/**
 * Public runtime configuration.
 *
 * NEXT_PUBLIC_* values are inlined into the browser bundle at *build* time (see the Next.js
 * environment-variables guide), so on Vercel they must be set before the build runs.
 * Reference the variable literally: `process.env[name]` would not be inlined.
 */

const DEFAULT_API_BASE_URL = "http://localhost:8000";

export const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_BASE_URL || DEFAULT_API_BASE_URL
).replace(/\/+$/, "");
