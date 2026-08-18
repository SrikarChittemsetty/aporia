/**
 * Client for the Aporia search API (`GET /search` in `api/main.py`).
 *
 * The base URL comes from NEXT_PUBLIC_APORIA_API. When it is unset — which is
 * the case for the public demo deployment — the app serves pre-computed results
 * from the bundled corpus instead. See `search()` in `search.ts` for how the
 * two paths are chosen, and why a rate-limit response is never quietly swapped
 * for demo data.
 */
import type { PassageContext, SearchResult } from "./types";

/** Empty string when unset, which `isLiveConfigured` treats as "no backend". */
const BASE = (process.env.NEXT_PUBLIC_APORIA_API ?? "").replace(/\/+$/, "");

/**
 * The first search on an unseen claim runs retrieval, a batched Claude call for
 * stance, and claim resolution — measured at roughly 30s cold. A default fetch
 * timeout would cut that off and report a network failure for a request that was
 * working fine, so the ceiling is set well above the observed worst case.
 */
const REQUEST_TIMEOUT_MS = 90_000;

export function isLiveConfigured(): boolean {
  return BASE.length > 0;
}

export function liveBaseUrl(): string {
  return BASE;
}

/** The API refused because this client is over its per-window budget. */
export class RateLimitError extends Error {
  readonly retryAfterSeconds: number;

  constructor(retryAfterSeconds: number, message: string) {
    super(message);
    this.name = "RateLimitError";
    this.retryAfterSeconds = retryAfterSeconds;
  }
}

/** The query itself was rejected — too short, too long, malformed. */
export class ValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ValidationError";
  }
}

/** The backend was unreachable or returned an error we can't attribute. */
export class BackendUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "BackendUnavailableError";
  }
}

/** FastAPI validation errors arrive as {detail: [{msg, loc, ...}]}. */
function readDetail(body: unknown, fallback: string): string {
  if (typeof body !== "object" || body === null) return fallback;
  const detail = (body as { detail?: unknown }).detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const first = detail[0] as { msg?: unknown } | undefined;
    if (first && typeof first.msg === "string") return first.msg;
  }
  return fallback;
}

/**
 * Hit the real backend. Throws a typed error rather than returning a
 * discriminated union, so callers can decide per failure kind whether falling
 * back to demo data is honest — for a rate limit it is not.
 */
export async function searchLive(query: string, k?: number): Promise<SearchResult> {
  if (!isLiveConfigured()) {
    throw new BackendUnavailableError("No API base URL is configured.");
  }

  const params = new URLSearchParams({ q: query });
  if (k !== undefined) params.set("k", String(k));

  let response: Response;
  try {
    response = await fetch(`${BASE}/search?${params}`, {
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
      headers: { Accept: "application/json" },
    });
  } catch (cause) {
    // Covers DNS failure, connection refused, CORS rejection and the timeout
    // above. They are indistinguishable from the browser and all mean the same
    // thing to a caller: the backend did not answer.
    const reason = cause instanceof Error && cause.name === "TimeoutError"
      ? `The backend did not respond within ${REQUEST_TIMEOUT_MS / 1000}s.`
      : "The backend could not be reached.";
    throw new BackendUnavailableError(reason);
  }

  if (response.status === 429) {
    // The server's own Retry-After is authoritative; the fallback only matters
    // if a proxy strips the header.
    const header = Number(response.headers.get("Retry-After"));
    const retryAfter = Number.isFinite(header) && header > 0 ? header : 60;
    const body = await response.json().catch(() => null);
    throw new RateLimitError(
      retryAfter,
      readDetail(body, "Too many searches. Each new claim costs a model call."),
    );
  }

  if (response.status === 422) {
    const body = await response.json().catch(() => null);
    throw new ValidationError(readDetail(body, "That query was rejected."));
  }

  if (!response.ok) {
    throw new BackendUnavailableError(`The backend returned ${response.status}.`);
  }

  return (await response.json()) as SearchResult;
}

/**
 * Fetch a passage together with its neighbours in the original work.
 *
 * Only meaningful against a live backend — the offline export ships each
 * passage in isolation, with no way to walk outward from it. Callers check
 * `isLiveConfigured()` before offering this.
 */
export async function fetchPassageContext(
  id: number,
  contextChunks = 2,
): Promise<PassageContext> {
  if (!isLiveConfigured()) {
    throw new BackendUnavailableError("No API base URL is configured.");
  }

  let response: Response;
  try {
    response = await fetch(`${BASE}/passage/${id}?context=${contextChunks}`, {
      signal: AbortSignal.timeout(15_000),
      headers: { Accept: "application/json" },
    });
  } catch {
    throw new BackendUnavailableError("Could not load the surrounding text.");
  }

  if (!response.ok) {
    throw new BackendUnavailableError(`The backend returned ${response.status}.`);
  }

  return (await response.json()) as PassageContext;
}
