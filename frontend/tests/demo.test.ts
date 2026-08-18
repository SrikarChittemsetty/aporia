import { afterEach, describe, expect, it, vi } from "vitest";
import { NoDemoMatchError, findClaim, searchDemo } from "@/lib/demo";
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

function claim(text: string, topic: string): DemoClaim {
  return {
    claim: text,
    topic,
    blurb: "",
    for: [passage(1)],
    against: [passage(2)],
    nuance: [],
  };
}

const CLAIMS = [
  claim("Humans have free will", "free will"),
  claim("God exists", "the existence of god"),
  claim("Happiness is the highest good", "happiness"),
];

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("findClaim", () => {
  it("matches the full claim text", () => {
    expect(findClaim("Humans have free will", CLAIMS)?.claim.topic).toBe("free will");
  });

  it("matches on the short topic form", () => {
    // A user typing "free will" should reach "Humans have free will" — the
    // claim is reachable by either surface.
    expect(findClaim("free will", CLAIMS)?.claim.topic).toBe("free will");
  });

  it("is case and punctuation insensitive", () => {
    expect(findClaim("GOD EXISTS!!!", CLAIMS)?.claim.topic).toBe("the existence of god");
  });

  it("returns null rather than the nearest claim when nothing is close", () => {
    // The important half of the behaviour. Returning an unrelated claim would
    // be worse than admitting the offline corpus doesn't cover the query,
    // because the user cannot tell a wrong answer from a right one here.
    expect(findClaim("quantum mechanics and wave collapse", CLAIMS)).toBeNull();
  });

  it("is not fooled by shared stopwords alone", () => {
    // Without stopword filtering, "the" and "is" make every claim a partial
    // match for every query.
    expect(findClaim("what is the thing", CLAIMS)).toBeNull();
  });
});

describe("searchDemo", () => {
  function stubCorpus(claims: DemoClaim[]) {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({ claims, stats: {} }),
      })),
    );
  }

  it("returns the claim's passages grouped by stance", async () => {
    stubCorpus(CLAIMS);
    const result = await searchDemo("free will");
    expect(result.claim).toBe("Humans have free will");
    expect(result.for).toHaveLength(1);
    expect(result.against).toHaveLength(1);
  });

  it("flags a topic query so the UI can explain the substitution", async () => {
    stubCorpus(CLAIMS);
    const result = await searchDemo("free will");
    // The user asked "free will" but is being shown a debate about "Humans
    // have free will" — For and Against are meaningless without saying so.
    expect(result.was_topic).toBe(true);
  });

  it("does not flag a query that already matches the claim", async () => {
    stubCorpus(CLAIMS);
    const result = await searchDemo("Humans have free will");
    expect(result.was_topic).toBe(false);
  });

  it("raises NoDemoMatchError carrying what it does have", async () => {
    stubCorpus(CLAIMS);
    await expect(searchDemo("quantum mechanics")).rejects.toBeInstanceOf(NoDemoMatchError);

    // The available claims ride along so the UI can offer them as suggestions
    // instead of leaving the user guessing what is covered.
    try {
      await searchDemo("quantum mechanics");
    } catch (err) {
      expect((err as NoDemoMatchError).available).toHaveLength(3);
    }
  });
});

describe("loadDemoData", () => {
  it("does not cache a failed load", async () => {
    // A rejected promise left in the cache would make one flaky network blip
    // permanently break the demo for the rest of the session.
    //
    // The cache is module-level, and the searchDemo tests above have already
    // filled it — so this needs a freshly imported module to observe a cold
    // start rather than their leftovers.
    vi.resetModules();
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({ ok: false, status: 500 })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ claims: CLAIMS, stats: {} }) });
    vi.stubGlobal("fetch", fetchMock);

    const { loadDemoData: freshLoad } = await import("@/lib/demo");

    await expect(freshLoad()).rejects.toThrow();
    // Second call must retry rather than replay the cached rejection.
    await expect(freshLoad()).resolves.toHaveProperty("claims");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});
