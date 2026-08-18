/**
 * Offline corpus: pre-computed results for the 12 claims in the demo export.
 *
 * These are not mock objects. Retrieval ran through the real from-scratch HNSW
 * index over bge-small embeddings, and the stance labels come from the same
 * Claude prompt the live pipeline uses — `scripts/export_site.py` recorded the
 * output. What is missing versus live is only the ability to answer a claim
 * nobody has run before.
 */
import type { DemoClaim, DemoData, SearchResult } from "./types";

/**
 * Fetched rather than imported: the export is ~230 KB, and bundling it would
 * put all of it in the initial JavaScript payload for a page most visitors
 * never search from. This way it is paid for on first use and cached after.
 */
let cache: Promise<DemoData> | null = null;

export function loadDemoData(): Promise<DemoData> {
  if (cache === null) {
    cache = fetch("./demo-data.json")
      .then((r) => {
        if (!r.ok) throw new Error(`demo-data.json returned ${r.status}`);
        return r.json() as Promise<DemoData>;
      })
      // Don't cache a rejected promise, or one flaky load would leave the demo
      // permanently broken for the rest of the session.
      .catch((err: unknown) => {
        cache = null;
        throw err;
      });
  }
  return cache;
}

/** Lowercase, drop punctuation, collapse whitespace. */
function normalize(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/**
 * Words carrying no topical signal. Without this, "is" and "the" make every
 * claim look like a partial match for every query.
 */
const STOPWORDS = new Set([
  "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does",
  "for", "from", "has", "have", "in", "is", "it", "its", "may", "must", "not",
  "of", "on", "only", "or", "that", "the", "there", "to", "was", "we", "what",
  "when", "which", "who", "will", "with",
]);

function contentWords(text: string): string[] {
  return normalize(text).split(" ").filter((w) => w.length > 2 && !STOPWORDS.has(w));
}

/**
 * Jaccard overlap on content words, plus a bonus when one string contains the
 * other outright. Deliberately simple — this only has to pick among 12 known
 * claims, and anything cleverer would imply a semantic capability the offline
 * build does not actually have.
 */
function score(query: string, candidate: string): number {
  const q = normalize(query);
  const c = normalize(candidate);
  if (q === c) return 1;

  const qWords = new Set(contentWords(query));
  const cWords = new Set(contentWords(candidate));
  if (qWords.size === 0 || cWords.size === 0) return 0;

  let shared = 0;
  for (const w of qWords) if (cWords.has(w)) shared += 1;
  const union = qWords.size + cWords.size - shared;
  const jaccard = union === 0 ? 0 : shared / union;

  const containment = c.includes(q) || q.includes(c) ? 0.4 : 0;
  return Math.min(1, jaccard + containment);
}

/**
 * Below this, a claim is a coincidence rather than a match. Tuned so that
 * "free will" hits "Humans have free will" but "quantum mechanics" hits
 * nothing at all — returning an unrelated claim would be worse than admitting
 * the offline corpus doesn't cover it.
 */
const MATCH_THRESHOLD = 0.25;

export interface DemoMatch {
  claim: DemoClaim;
  score: number;
}

export function findClaim(query: string, claims: DemoClaim[]): DemoMatch | null {
  let best: DemoMatch | null = null;
  for (const claim of claims) {
    // A claim is reachable by either its full statement or its short topic.
    const s = Math.max(score(query, claim.claim), score(query, claim.topic));
    if (best === null || s > best.score) best = { claim, score: s };
  }
  return best !== null && best.score >= MATCH_THRESHOLD ? best : null;
}

/** Raised when the offline corpus has nothing for this query. */
export class NoDemoMatchError extends Error {
  readonly available: DemoClaim[];

  constructor(available: DemoClaim[]) {
    super("The offline demo corpus does not cover that claim.");
    this.name = "NoDemoMatchError";
    this.available = available;
  }
}

export async function searchDemo(query: string): Promise<SearchResult> {
  const data = await loadDemoData();
  const match = findClaim(query, data.claims);
  if (match === null) throw new NoDemoMatchError(data.claims);

  const { claim } = match;
  return {
    query,
    claim: claim.claim,
    // The banner explaining "you asked X, we're debating Y" should appear
    // whenever the two differ, exactly as it would for a live topic query.
    was_topic: normalize(query) !== normalize(claim.claim),
    stance_error: null,
    for: claim.for,
    against: claim.against,
    nuance: claim.nuance,
  };
}
