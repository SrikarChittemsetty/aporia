import type { Passage, Stance } from "@/lib/types";
import PassageCard from "./PassageCard";
import { STANCE_STYLES } from "./stance-style";

interface Props {
  stance: Stance;
  passages: Passage[];
}

export default function StanceColumn({ stance, passages }: Props) {
  const style = STANCE_STYLES[stance];

  return (
    <section aria-labelledby={`heading-${stance}`}>
      <h2
        id={`heading-${stance}`}
        className={`mb-4 flex items-center gap-2 text-[1.05rem] uppercase tracking-[0.12em] ${style.heading}`}
      >
        <span className={`inline-block h-2 w-2 rounded-full ${style.dot}`} aria-hidden="true" />
        {style.label}
        <span className="text-muted normal-case tracking-normal">({passages.length})</span>
      </h2>

      {passages.length === 0 ? (
        // An empty side is a real finding — the corpus genuinely has nobody
        // arguing that way — so it gets stated rather than left blank.
        <p className="rounded-lg border border-dashed border-line px-4 py-6 text-center text-[0.9rem] text-muted">
          Nothing found on this side.
        </p>
      ) : (
        <div className="space-y-4">
          {passages.map((passage) => (
            <PassageCard key={passage.id} passage={passage} stance={stance} />
          ))}
        </div>
      )}
    </section>
  );
}
