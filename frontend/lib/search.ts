/**
 * Chooses between the live API and the offline corpus, and decides what the
 * user is told about which one answered.
 *
 * The rule this file exists to enforce: the UI never presents demo data as
 * though it came from the live backend. Degrading quietly would make the demo
 * look like it always works, which is the opposite of what the project is
 * trying to demonstrate.
 */
import {
  BackendUnavailableError,
  RateLimitError,
  ValidationError,
  isLiveConfigured,
  searchLive,
} from "./api";
import { NoDemoMatchError, searchDemo } from "./demo";
import type { SearchResponse } from "./types";

export { NoDemoMatchError, RateLimitError, ValidationError, BackendUnavailableError };

export async function search(query: string, k?: number): Promise<SearchResponse> {
  const trimmed = query.trim();
  if (trimmed.length < 3) {
    throw new ValidationError("Give me at least three characters to work with.");
  }

  if (!isLiveConfigured()) {
    // Deployed without a backend. This is the expected state for the public
    // demo, so it is a plain fact in the UI rather than a warning.
    return {
      result: await searchDemo(trimmed),
      source: "demo",
      notice: "Showing pre-computed results from the offline corpus.",
    };
  }

  try {
    return { result: await searchLive(trimmed, k), source: "live" };
  } catch (error) {
    // A rate limit is the backend working correctly. Swapping in demo data here
    // would hide the limiter and teach the user that the quota isn't real, so
    // it propagates and the UI shows the countdown.
    if (error instanceof RateLimitError) throw error;

    // Likewise a rejected query: answering it from the demo corpus would imply
    // the backend accepted something it did not.
    if (error instanceof ValidationError) throw error;

    if (error instanceof BackendUnavailableError) {
      try {
        return {
          result: await searchDemo(trimmed),
          source: "demo",
          notice: `${error.message} Showing pre-computed results instead.`,
        };
      } catch (fallbackError) {
        // Backend down *and* nothing offline for this query. Report the real
        // cause — the outage — rather than the fallback's miss, which would
        // point the user at the wrong problem.
        if (fallbackError instanceof NoDemoMatchError) throw error;
        throw fallbackError;
      }
    }

    throw error;
  }
}
