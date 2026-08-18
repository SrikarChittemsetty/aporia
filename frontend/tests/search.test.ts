/**
 * The rules in lib/search.ts about when demo data may stand in for a live
 * answer — and, more importantly, when it may not.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { DemoClaim, Passage } from "@/lib/types";

function passage(id: number): Passage {
  return {
    id,
    author: "Author",
    work: "Work",
    citation: "I.1",
    text: "text",
    move: "move",
    confidence: 0.9,
    similarity: 0.5,
  };
}

const CLAIMS: DemoClaim[] = [
  {
    claim: "Humans have free will",
    topic: "free will",
    blurb: "",
    for: [passage(1)],
    against: [passage(2)],
    nuance: [],
  },
];

/**
 * `lib/api.ts` reads NEXT_PUBLIC_APORIA_API once at module load, so the env has
 * to be set before the import — hence resetModules and a dynamic import rather
 * than a top-level one.
 */
async function loadSearch(apiBase: string | undefined) {
  vi.resetModules();
  if (apiBase === undefined) {
    vi.stubEnv("NEXT_PUBLIC_APORIA_API", "");
  } else {
    vi.stubEnv("NEXT_PUBLIC_APORIA_API", apiBase);
  }
  return import("@/lib/search");
}

/** fetch stub that serves the demo corpus and delegates /search to `onSearch`. */
function stubFetch(onSearch: (url: string) => unknown) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      if (url.includes("demo-data.json")) {
        return { ok: true, json: async () => ({ claims: CLAIMS, stats: {} }) };
      }
      return onSearch(url);
    }),
  );
}

function liveResult() {
  return {
    ok: true,
    status: 200,
    json: async () => ({
      query: "free will",
      claim: "Humans have free will",
      was_topic: false,
      stance_error: null,
      for: [passage(1)],
      against: [],
      nuance: [],
    }),
  };
}

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn());
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("with no backend configured", () => {
  it("answers from the offline corpus and says so", async () => {
    const { search } = await loadSearch(undefined);
    stubFetch(() => {
      throw new Error("should not have called the API");
    });

    const response = await search("free will");
    expect(response.source).toBe("demo");
    expect(response.notice).toBeTruthy();
    expect(response.result.claim).toBe("Humans have free will");
  });
});

describe("with a backend configured", () => {
  it("returns the live result unlabelled by any notice", async () => {
    const { search } = await loadSearch("http://api.test");
    stubFetch(() => liveResult());

    const response = await search("free will");
    expect(response.source).toBe("live");
    expect(response.notice).toBeUndefined();
  });

  it("falls back to demo data when the backend is unreachable, and says why", async () => {
    const { search } = await loadSearch("http://api.test");
    stubFetch(() => {
      throw new TypeError("connection refused");
    });

    const response = await search("free will");
    expect(response.source).toBe("demo");
    // Silently serving stale data as though it were live is the failure this
    // whole module exists to avoid.
    expect(response.notice).toMatch(/could not be reached/i);
  });

  it("does NOT swap demo data in for a rate limit", async () => {
    // The single most important rule here. Falling back would hide the limiter
    // and teach the user the quota isn't real.
    const { search, RateLimitError } = await loadSearch("http://api.test");
    stubFetch(() => ({
      ok: false,
      status: 429,
      headers: { get: (h: string) => (h === "Retry-After" ? "42" : null) },
      json: async () => ({ detail: "Too many searches." }),
    }));

    await expect(search("free will")).rejects.toBeInstanceOf(RateLimitError);
  });

  it("surfaces the server's Retry-After rather than guessing", async () => {
    const { search, RateLimitError } = await loadSearch("http://api.test");
    stubFetch(() => ({
      ok: false,
      status: 429,
      headers: { get: (h: string) => (h === "Retry-After" ? "42" : null) },
      json: async () => ({ detail: "Too many searches." }),
    }));

    try {
      await search("free will");
      expect.unreachable("should have thrown");
    } catch (err) {
      expect(err).toBeInstanceOf(RateLimitError);
      expect((err as InstanceType<typeof RateLimitError>).retryAfterSeconds).toBe(42);
    }
  });

  it("does NOT swap demo data in for a rejected query", async () => {
    // Answering a 422 from the corpus would imply the backend accepted
    // something it explicitly refused.
    const { search, ValidationError } = await loadSearch("http://api.test");
    stubFetch(() => ({
      ok: false,
      status: 422,
      headers: { get: () => null },
      json: async () => ({ detail: [{ msg: "String should have at most 300 characters" }] }),
    }));

    await expect(search("free will")).rejects.toBeInstanceOf(ValidationError);
  });

  it("reports the outage, not the corpus miss, when both fail", async () => {
    // Backend down AND nothing offline for this query. Reporting "not in the
    // demo corpus" would point the user at the wrong problem entirely.
    const { search, BackendUnavailableError } = await loadSearch("http://api.test");
    stubFetch(() => {
      throw new TypeError("connection refused");
    });

    await expect(search("quantum mechanics")).rejects.toBeInstanceOf(
      BackendUnavailableError,
    );
  });
});

describe("input bounds", () => {
  it("rejects a query too short to be meaningful before any network call", async () => {
    const { search, ValidationError } = await loadSearch("http://api.test");
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    await expect(search("ab")).rejects.toBeInstanceOf(ValidationError);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
