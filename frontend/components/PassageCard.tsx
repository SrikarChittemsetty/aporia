"use client";

import { useState } from "react";
import { fetchPassageContext, isLiveConfigured } from "@/lib/api";
import type { ContextChunk, Passage, Stance } from "@/lib/types";
import { STANCE_STYLES } from "./stance-style";

/** Passages run long. Show enough to judge relevance, expand for the rest. */
const PREVIEW_CHARS = 520;

interface Props {
  passage: Passage;
  stance: Stance;
}

export default function PassageCard({ passage, stance }: Props) {
  const style = STANCE_STYLES[stance];
  const [expanded, setExpanded] = useState(false);
  const [context, setContext] = useState<ContextChunk[] | null>(null);
  const [contextState, setContextState] = useState<"idle" | "loading" | "error">("idle");

  const isLong = passage.text.length > PREVIEW_CHARS;
  const preview = isLong ? `${passage.text.slice(0, PREVIEW_CHARS).trimEnd()}…` : passage.text;

  async function loadContext() {
    setContextState("loading");
    try {
      const data = await fetchPassageContext(passage.id);
      setContext(data.context);
      setContextState("idle");
    } catch {
      // Surfaced in place rather than thrown: failing to widen the view of one
      // passage shouldn't tear down a page of results that are otherwise fine.
      setContextState("error");
    }
  }

  return (
    <article
      className={`animate-fade-in rounded-lg border border-line border-l-4 bg-card p-4 sm:p-5 ${style.edge}`}
    >
      {passage.move && (
        <p className={`mb-3 rounded-md px-3 py-2 text-[0.92rem] leading-snug ${style.move}`}>
          {passage.move}
        </p>
      )}

      {context ? (
        <div className="mb-3 space-y-3 text-[0.98rem] leading-relaxed">
          {context.map((chunk) => (
            <p
              key={chunk.id}
              // The passage that actually matched stays emphasised, so widening
              // the view doesn't lose track of which chunk was retrieved.
              className={chunk.id === passage.id ? "text-ink" : "text-muted"}
            >
              {chunk.text}
            </p>
          ))}
        </div>
      ) : (
        <blockquote className="mb-3 text-[0.98rem] leading-relaxed">
          &ldquo;{expanded ? passage.text : preview}&rdquo;
        </blockquote>
      )}

      <p className="text-[0.85rem] text-muted">
        <span className="font-semibold text-ink">{passage.author}</span>
        {", "}
        <em>{passage.work}</em>
        {passage.citation ? ` — ${passage.citation}` : ""}
      </p>

      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[0.78rem] text-muted">
        <span title="Cosine similarity between the search vector and this passage">
          similarity {passage.similarity.toFixed(3)}
        </span>
        {passage.confidence > 0 && (
          <span title="How confident the stance classifier was in this label">
            confidence {passage.confidence.toFixed(2)}
          </span>
        )}

        {isLong && !context && (
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            className="text-accent underline decoration-dotted underline-offset-2 hover:opacity-80"
          >
            {expanded ? "show less" : "show full passage"}
          </button>
        )}

        {/* The surrounding-text endpoint exists only on the live backend. */}
        {isLiveConfigured() && !context && contextState !== "loading" && (
          <button
            type="button"
            onClick={loadContext}
            className="text-accent underline decoration-dotted underline-offset-2 hover:opacity-80"
          >
            read in context
          </button>
        )}
        {contextState === "loading" && <span aria-live="polite">loading context…</span>}
        {contextState === "error" && (
          <span className="text-against" role="status">
            couldn&rsquo;t load context
          </span>
        )}
      </div>
    </article>
  );
}
