"""A held-out query set, written after the fix and never used to build it.

evals/sanity.py is where the retrieval failure was *found*, so it is not a fair
place to prove the failure was fixed: query expansion could score 41/41 there by
happening to patch four known holes. The number that means something comes from
queries the method has never seen.

These twenty were written fresh against the corpus contents, in the same register
as the originals — a claim as someone would actually type it, not a phrase lifted
from the text — and deliberately including several that state a conclusion in
modern abstract terms, since that is the failure mode under test. No query here
was checked against retrieval before being written down, and none was revised
after seeing a result.

    python -m evals.expansion_eval --holdout
"""

SPINOZA = "Baruch Spinoza"
HUME = "David Hume"
NIETZSCHE = "Friedrich Nietzsche"
KANT = "Immanuel Kant"
LOCKE = "John Locke"
MILL = "John Stuart Mill"
PLATO = "Plato"
DESCARTES = "René Descartes"
PAINE = "Thomas Paine"
JAMES = "William James"

QUERIES: list[tuple[str, set[str], str]] = [
    ("the soul is immortal and survives the death of the body", {PLATO}, "justice"),
    ("a society is only as just as the way it treats its weakest members", {PLATO, MILL}, "justice"),
    ("education should turn the mind toward what is real rather than what merely appears",
     {PLATO}, "justice"),
    ("wealth corrupts the character of those who rule", {PLATO}, "justice"),

    ("everything that exists follows necessarily from the nature of what came before",
     {SPINOZA}, "free will"),
    ("emotions enslave us until we understand their causes", {SPINOZA}, "free will"),
    ("God and nature are two names for the same thing", {SPINOZA}, "god"),

    ("we can never observe a necessary connection between two events", {HUME}, "knowledge"),
    ("the mind is a bundle of impressions with no underlying self", {HUME}, "knowledge"),
    ("a wise person proportions belief to the evidence", {HUME}, "knowledge"),

    ("certainty must be rebuilt from a single indubitable foundation", {DESCARTES}, "knowledge"),
    ("the mind and the body are entirely different kinds of thing", {DESCARTES}, "knowledge"),

    ("a good will is the only thing good without qualification", {KANT}, "morality"),
    ("using a person merely as a tool for your own ends is the root of immorality",
     {KANT}, "morality"),

    ("pity and humility are symptoms of decline rather than virtues", {NIETZSCHE}, "morality"),
    ("what a philosopher calls truth is usually his own temperament in disguise",
     {NIETZSCHE}, "morality"),

    ("eccentricity is valuable because conformity makes a society stagnant", {MILL}, "politics"),
    ("an unpopular opinion may turn out to be the true one", {MILL}, "politics"),

    ("no man may be judge in his own case", {LOCKE}, "politics"),
    ("the accident of birth confers no right to govern others", {PAINE}, "politics"),
    ("a live option that cannot wait for proof must be decided by the will",
     {JAMES}, "god"),
]
