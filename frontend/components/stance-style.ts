import type { Stance } from "@/lib/types";

/**
 * Tailwind scans source files for complete class strings, so a template like
 * `text-${stance}` would compile to nothing at build time. Every class a stance
 * can take is written out here in full.
 */
export interface StanceStyle {
  label: string;
  heading: string;
  /** Applied to the card's leading edge so sides are distinguishable at a glance. */
  edge: string;
  /** The "argumentative move" strip inside each card. */
  move: string;
  dot: string;
}

export const STANCE_STYLES: Record<Stance, StanceStyle> = {
  for: {
    label: "For",
    heading: "text-for",
    edge: "border-l-for",
    move: "bg-for-soft text-for",
    dot: "bg-for",
  },
  against: {
    label: "Against",
    heading: "text-against",
    edge: "border-l-against",
    move: "bg-against-soft text-against",
    dot: "bg-against",
  },
  nuance: {
    label: "Nuance & reframings",
    heading: "text-nuance",
    edge: "border-l-nuance",
    move: "bg-nuance-soft text-nuance",
    dot: "bg-nuance",
  },
};
