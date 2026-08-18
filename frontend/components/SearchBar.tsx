"use client";

/** Mirrors MAX_QUERY_CHARS in api/limits.py — the server rejects longer. */
const MAX_QUERY_CHARS = 300;

/**
 * Every one of these resolves in the offline corpus as well as live, so the
 * suggested queries work on the public demo rather than dead-ending on
 * "not covered".
 */
const EXAMPLES = [
  "free will",
  "God exists",
  "morality is objective",
  "happiness",
  "personal identity",
];

interface Props {
  value: string;
  onChange: (value: string) => void;
  onSearch: (query: string) => void;
  busy: boolean;
}

/**
 * Fully controlled. An earlier version kept its own copy of the query and
 * needed an effect to re-sync when a ?q= link loaded — two sources of truth for
 * one value. The page owns it now, so there is nothing to synchronise.
 */
export default function SearchBar({ value, onChange, onSearch, busy }: Props) {
  function submit(query: string) {
    const trimmed = query.trim();
    if (trimmed.length < 3 || busy) return;
    onSearch(trimmed);
  }

  const remaining = MAX_QUERY_CHARS - value.length;

  return (
    <div className="mx-auto w-full max-w-2xl">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit(value);
        }}
        className="flex gap-2"
      >
        <label htmlFor="query" className="sr-only">
          A topic or a claim
        </label>
        <input
          id="query"
          type="text"
          value={value}
          maxLength={MAX_QUERY_CHARS}
          onChange={(e) => onChange(e.target.value)}
          placeholder='a topic ("free will") or a claim ("God exists")'
          autoComplete="off"
          className="flex-1 rounded-lg border border-line bg-card px-4 py-3 text-[1.05rem] placeholder:text-muted focus:border-accent focus:outline-none"
        />
        <button
          type="submit"
          disabled={busy || value.trim().length < 3}
          className="rounded-lg bg-ink px-5 py-3 text-[1rem] text-canvas transition-opacity disabled:cursor-not-allowed disabled:opacity-50"
        >
          {busy ? "Searching…" : "Search"}
        </button>
      </form>

      <div className="mt-3 flex flex-wrap items-center justify-center gap-x-2 gap-y-1 text-[0.9rem] text-muted">
        <span>Try:</span>
        {EXAMPLES.map((example) => (
          <button
            key={example}
            type="button"
            disabled={busy}
            onClick={() => {
              onChange(example);
              submit(example);
            }}
            className="text-accent underline decoration-dotted underline-offset-2 hover:opacity-80 disabled:opacity-50"
          >
            {example}
          </button>
        ))}
      </div>

      {/* Only appears once it's nearly relevant — a counter sitting at 300
          from the start is noise. */}
      {remaining < 60 && (
        <p className="mt-2 text-center text-[0.8rem] text-muted" aria-live="polite">
          {remaining} characters left
        </p>
      )}
    </div>
  );
}
