/**
 * Types mirroring the Python API in `api/main.py`.
 *
 * The bundled demo data (`public/demo-data.json`, generated from the same
 * pipeline by `scripts/export_site.py`) has an identical passage shape, which
 * is why one set of types serves both the live and offline paths.
 */

/** Which side of the claim a passage argues. Matches the API's grouped keys. */
export type Stance = "for" | "against" | "nuance";

export const STANCES: readonly Stance[] = ["for", "against", "nuance"] as const;

export interface Passage {
  id: number;
  author: string;
  work: string;
  /** Where in the work, e.g. "Treatise II.3.3". */
  citation: string;
  text: string;
  /** One-line summary of the argumentative move the passage makes. */
  move: string;
  /** Stance classifier's confidence, 0–1. */
  confidence: number;
  /** Cosine similarity to the search vector, 0–1. */
  similarity: number;
}

export interface SearchResult {
  query: string;
  /** The canonical claim being debated. Differs from `query` for bare topics. */
  claim: string;
  /** True when the query was a topic ("free will") rather than a claim. */
  was_topic: boolean;
  /**
   * Set when the stance layer degraded — no LLM backend reachable, so passages
   * come back unclassified. The API deliberately returns results anyway rather
   * than failing the whole request.
   */
  stance_error: string | null;
  for: Passage[];
  against: Passage[];
  nuance: Passage[];
}

/** One chunk of surrounding text, from `GET /passage/{id}`. */
export interface ContextChunk {
  id: number;
  citation_path: string;
  seq: number;
  text: string;
}

export interface PassageContext {
  id: number;
  author: string;
  work: string;
  citation: string;
  /** The passage plus its neighbours in the original work, in reading order. */
  context: ContextChunk[];
}

/** Where a result came from. Surfaced in the UI — never hidden from the user. */
export type ResultSource = "live" | "demo";

export interface SearchResponse {
  result: SearchResult;
  source: ResultSource;
  /**
   * Human-readable explanation when the answer isn't what was asked for —
   * e.g. the live API was unreachable and this is bundled data instead.
   */
  notice?: string;
}

/**
 * A pre-computed claim in the offline demo corpus.
 *
 * Deliberately NOT a `SearchResult`: the export has no `query`, `was_topic` or
 * `stance_error`, because those describe one request rather than the claim
 * itself. `demo.ts` supplies them when it turns a claim into a result.
 */
export interface DemoClaim {
  claim: string;
  /** Short topic form, e.g. "free will" — matched against loosely. */
  topic: string;
  /** One-line editorial summary of the disagreement, shown on the landing page. */
  blurb: string;
  for: Passage[];
  against: Passage[];
  nuance: Passage[];
}

export interface DemoStats {
  passages: number;
  works: number;
  authors: number;
  claims: number;
  classified: number;
  for: number;
  against: number;
  nuance: number;
}

export interface DemoData {
  claims: DemoClaim[];
  stats: DemoStats;
}
