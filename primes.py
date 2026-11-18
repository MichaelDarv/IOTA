#!/usr/bin/env python3
"""
nsm_fitness.py - a standalone fitness function for "English -> 65 semantic primes".

    fitness = validity_ratio ** gamma * faithfulness * (1 if fully valid else invalid_penalty)

  validity     Is the candidate written ONLY in the 65 primes?
               (deterministic, offline, no dependencies)
  faithfulness Round trip. A frozen back-translator sees ONLY the prime text and
               rewrites it as English; we score how close that is to the source.
               (needs a back-translator you supply; see adapters below)

Why multiply? (The invalid_penalty is a gate: ANY non-prime token cuts the score hard,
because downstream processing needs prime-only text.) A candidate that copies the source sentence verbatim would round-trip
perfectly but is not prime-only, so validity drives its score to ~0. A candidate that
is perfectly prime-only but says something else gets ~0 faithfulness. Only candidates
that are both score well.

Quick use
    from nsm_fitness import fitness, anthropic_backtranslator
    bt = anthropic_backtranslator()
    r = fitness("The dog died.", "BEFORE NOW, SOMETHING THAT LIVES DIED", bt)
    print(r.fitness, r.validity, r.faithfulness, r.invalid_tokens)

CLI
    python nsm_fitness.py --check "SOMEONE DIED"     validity only (offline)
    python nsm_fitness.py --check "SOMEONE DIED" --strict
    python nsm_fitness.py --selftest                 run built-in tests (offline, mock back-translator)
"""
from __future__ import annotations

import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Optional

# ==========================================================================
# 1. VALIDITY: is the text made only of the 65 primes?
# ==========================================================================

# Single-word realisations of the primes (canonical NSM English exponents).
BASE = set("""
i you someone something thing people body kind part this same other else another
one two much many little few some all good bad big small
think know want feel see hear say words word true do happen move touch
have live die when time now before after moment where place here above below far
near side inside not maybe can because if very more like way as
""".split())
# 'my'/'your'/'these' are standard inflected forms of I / YOU / THIS; BE has allolexes.
FORMS_OF_BASE = {"my", "your", "these", "be", "am", "is", "are", "was", "were", "been", "being"}

# Inflected forms (hand-listed on purpose: stripping suffixes would let "seed" pass as "see").
INFLECTIONS = {}
for _lemma, _forms in {
    "think": "thinks thinking thought", "know": "knows knowing knew known",
    "want": "wants wanted wanting", "feel": "feels feeling felt",
    "see": "sees seeing saw seen", "hear": "hears hearing heard",
    "say": "says saying said", "do": "does doing did done",
    "happen": "happens happened happening", "move": "moves moved moving",
    "touch": "touches touched touching", "have": "has having had",
    "live": "lives lived living", "die": "dies died dying",
    "thing": "things", "word": "words", "kind": "kinds", "part": "parts",
    "place": "places", "time": "times", "side": "sides", "moment": "moments",
    "body": "bodies", "one": "ones",
}.items():
    for _f in _forms.split():
        INFLECTIONS[_f] = _lemma

# Multi-word primes. Words like LONG, SHORT, THERE and FOR are only valid inside these.
PHRASES = sorted([
    ("for", "a", "long", "time"), ("for", "a", "short", "time"), ("for", "some", "time"),
    ("a", "long", "time"), ("a", "short", "time"),
    ("there", "is"), ("there", "are"), ("there", "was"), ("there", "were"),
    ("the", "same"), ("a", "moment"), ("a", "few"), ("a", "little"),
], key=len, reverse=True)

# Grammar glue tolerated in lenient mode (counted invalid in strict mode).
GRAMMAR = {"a", "an", "the", "to", "of", "it", "its", "they", "them", "their", "that",
           "what", "who", "which", "with", "for", "from", "at", "by", "than", "and", "or", "on"}

CONTRACTION_SUFFIX = {"re": "are", "ve": "have", "ll": "will", "m": "am", "d": "would"}
TOKEN_RE = re.compile(r"«[^»]*»|\[[^\]]*\]|\d+|[A-Za-z]+(?:'[A-Za-z]+)?")


def _tokenize(text: str):
    """Return a list of (surface, lowercase-word-or-marker) pairs."""
    out = []
    for m in TOKEN_RE.finditer(text):
        raw = m.group(0)
        if raw[0] in "«[" or raw[0].isdigit():
            out.append((raw, raw))
            continue
        low = raw.lower()
        if low in ("won't",):
            out += [(raw, "will"), (raw, "not")]
        elif low == "can't":
            out += [(raw, "can"), (raw, "not")]
        elif low.endswith("n't"):
            out += [(raw, low[:-3]), (raw, "not")]
        elif low.endswith("'s"):
            out.append((raw, low[:-2]))                 # possessive: classify the base
        elif "'" in low and low.split("'")[1] in CONTRACTION_SUFFIX:
            a, b = low.split("'")
            out += [(raw, a), (raw, CONTRACTION_SUFFIX[b])]
        else:
            out.append((raw, low))
    return out


def _is_prime_word(w: str) -> bool:
    return w in BASE or w in FORMS_OF_BASE or w in INFLECTIONS


@dataclass
class ValidityResult:
    ratio: float                     # fraction of tokens that are valid (0..1)
    all_valid: bool                  # True iff there are no invalid tokens
    n_tokens: int
    invalid_tokens: list = field(default_factory=list)
    grammar_tokens: list = field(default_factory=list)   # tolerated in lenient mode
    names: list = field(default_factory=list)


def check_validity(text: str, strict: bool = False, allow_names: bool = True) -> ValidityResult:
    """
    Score how much of `text` is expressed in primes.

    strict=False: small grammar words (a, the, to, of, that ...) are tolerated and reported.
    strict=True : only primes (and prime phrases) are valid.
    allow_names : «Maria»-style spans are passed through as opaque names.
    """
    toks = _tokenize(text)
    kinds = [None] * len(toks)
    words = [w for _, w in toks]

    # 1) multi-word primes
    i = 0
    while i < len(words):
        for ph in PHRASES:
            if tuple(words[i:i + len(ph)]) == ph:
                for k in range(i, i + len(ph)):
                    kinds[k] = "prime"
                i += len(ph) - 1
                break
        i += 1

    # 2) everything else, token by token
    invalid, grammar, names = [], [], []
    for k, (raw, w) in enumerate(toks):
        if kinds[k]:
            continue
        if w.startswith("«"):
            kinds[k] = "name" if allow_names else "invalid"
        elif w.startswith("[") or w[0].isdigit():
            kinds[k] = "invalid"
        elif _is_prime_word(w):
            kinds[k] = "prime"
        elif w in GRAMMAR:
            kinds[k] = "invalid" if strict else "grammar"
        else:
            kinds[k] = "invalid"
        if kinds[k] == "invalid":
            invalid.append(raw)
        elif kinds[k] == "grammar":
            grammar.append(raw)
        elif kinds[k] == "name":
            names.append(raw)

    n = len(toks)
    n_valid = n - len(invalid)
    return ValidityResult(
        ratio=(n_valid / n) if n else 0.0,
        all_valid=(n > 0 and not invalid),
        n_tokens=n, invalid_tokens=invalid, grammar_tokens=grammar, names=names,
    )


# ==========================================================================
# 2. FAITHFULNESS: round trip  English -> primes -> English, compare to source
# ==========================================================================

BackTranslator = Callable[[str], str]        # prime text -> English (must NOT see the source!)
Similarity = Callable[[str, str], float]     # (source, back-translation) -> 0..1

_STOP = set("a an the to of is are am was were be been being that and or in on at for from "
            "by with it its do does did".split())


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def lexical_similarity(a: str, b: str) -> float:
    """Dependency-free similarity: stemmed content words + character trigrams, 50/50.
    Cheap and good enough for testing the loop. For real training swap in
    embedding_similarity() or an NLI model, because lexical overlap under-rates paraphrase."""
    def prep(t):
        ws = [_stem(w) for w in re.findall(r"[a-z]+", t.lower()) if w not in _STOP]
        return ws
    wa, wb = prep(a), prep(b)
    word_sim = _cosine(Counter(wa), Counter(wb))
    tri = lambda ws: Counter(g for w in ws for g in (f"^{w}$"[i:i + 3] for i in range(max(1, len(w) - 1))))
    char_sim = _cosine(tri(wa), tri(wb))
    return 0.5 * word_sim + 0.5 * char_sim


# ---- optional adapters (imported lazily; not exercised by the self-test) -------------

def anthropic_backtranslator(model: str = "claude-sonnet-5-5", max_tokens: int = 300) -> BackTranslator:
    """Back-translator using the Anthropic API (needs `pip install anthropic` and an API key).
    The model sees ONLY the prime text, never the original sentence."""
    import anthropic
    client = anthropic.Anthropic()
    system = ("The input is written in the Natural Semantic Metalanguage (the 65 semantic primes). "
              "Rewrite it as plain, natural English that says exactly what it says. "
              "Do not add information. Output only the English.")

    @lru_cache(maxsize=None)
    def bt(prime_text: str) -> str:
        msg = client.messages.create(model=model, max_tokens=max_tokens, system=system,
                                     messages=[{"role": "user", "content": prime_text}])
        return msg.content[0].text.strip()
    return bt


def embedding_similarity(model_name: str = "all-MiniLM-L6-v2") -> Similarity:
    """Sentence-embedding cosine (needs `pip install sentence-transformers`)."""
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer(model_name)

    def sim(a: str, b: str) -> float:
        ea, eb = model.encode([a, b], convert_to_tensor=True)
        return max(0.0, float(util.cos_sim(ea, eb)))
    return sim


# ==========================================================================
# 3. THE FITNESS FUNCTION
# ==========================================================================

@dataclass
class FitnessResult:
    fitness: float
    validity: float
    faithfulness: float
    all_valid: bool
    back_translation: str
    invalid_tokens: list
    grammar_tokens: list


def fitness(source: str, prime_text: str, backtranslate: BackTranslator,
            similarity: Similarity = lexical_similarity, *,
            strict: bool = False, gamma: float = 2.0, invalid_penalty: float = 0.1,
            allow_names: bool = True) -> FitnessResult:
    """
    Score one candidate translation. Returns a FitnessResult; use `.fitness` as the reward.

    gamma > 1 makes invalid tokens costly (validity ** gamma). Raise it to push the model
    harder toward fully valid output; lower it if training stalls on early invalid outputs.
    invalid_penalty is a multiplier applied when ANY token is invalid (1.0 disables the gate).
    Lower values make "mostly valid" candidates much worse than "fully valid" ones, which stops a
    model from winning by leaking source words into the output.
    """
    v = check_validity(prime_text, strict=strict, allow_names=allow_names)
    if v.ratio == 0.0:                     # nothing valid: skip the (possibly paid) round trip
        return FitnessResult(0.0, 0.0, 0.0, False, "", v.invalid_tokens, v.grammar_tokens)

    back = backtranslate(prime_text)       # sees ONLY the prime text
    faith = max(0.0, min(1.0, similarity(source, back)))
    return FitnessResult(
        fitness=(v.ratio ** gamma) * faith * (1.0 if v.all_valid else invalid_penalty),
        validity=v.ratio, faithfulness=faith, all_valid=v.all_valid,
        back_translation=back, invalid_tokens=v.invalid_tokens, grammar_tokens=v.grammar_tokens,
    )


def fitness_batch(pairs, backtranslate: BackTranslator, **kw):
    """pairs: iterable of (source, candidate). Returns a list of FitnessResult."""
    return [fitness(s, c, backtranslate, **kw) for s, c in pairs]


# ==========================================================================
# 4. Self-test and CLI
# ==========================================================================

def _selftest() -> int:
    ok = True

    def expect(name, cond):
        nonlocal ok
        print(("  ok   " if cond else "  FAIL ") + name)
        ok &= bool(cond)

    print("Validity")
    v = check_validity("BEFORE NOW, SOMEONE DIED")
    expect("pure prime text is fully valid", v.all_valid and v.ratio == 1.0)
    v = check_validity("SOMEONE DOES NOT WANT THIS; BECAUSE OF THIS, SOMETHING BAD HAPPENED")
    expect("inflections (does, happened) accepted", v.all_valid)
    v = check_validity("I DON'T WANT THIS")
    expect("DON'T WANT expands to DO NOT WANT", v.all_valid)
    v = check_validity("SOMEONE WAS HERE FOR A LONG TIME")
    expect("'for a long time' accepted as a phrase", v.all_valid)
    v = check_validity("IT WAS A LONG WAY, SHORT PEOPLE")
    expect("'long'/'short' rejected outside their phrases", set(v.invalid_tokens) == {"LONG", "SHORT"} or "SHORT" in v.invalid_tokens)
    v = check_validity("THIS SOMEONE KILLS A DOG")
    expect("non-prime words flagged (kills, dog)", v.invalid_tokens == ["KILLS", "DOG"] and not v.all_valid)
    v = check_validity("A KIND OF SOMETHING THAT LIVES")
    expect("lenient mode tolerates a/of/that", v.all_valid and set(w.lower() for w in v.grammar_tokens) == {"a", "of", "that"})
    v = check_validity("A KIND OF SOMETHING THAT LIVES", strict=True)
    expect("strict mode rejects a/of/that", not v.all_valid)
    v = check_validity("SOMEONE «Maria» FELT SOMETHING GOOD")
    expect("names pass through when allowed", v.all_valid and v.names == ["«Maria»"])
    v = check_validity("seed feed")
    expect("'seed' is not mistaken for SEE", v.ratio == 0.0)
    v = check_validity("[dog?] SOMEONE")
    expect("unknown-word markers are invalid", v.invalid_tokens == ["[dog?]"])
    expect("empty text scores 0", check_validity("").ratio == 0.0)

    print("\nFitness (mock back-translator; the arithmetic is real, the back-translations are canned)")
    canned = {
        "BEFORE NOW, SOMETHING THAT LIVED DIED": "Before now, something that lived died.",
        "BEFORE NOW, SOMEONE FELT SOMETHING GOOD": "Someone felt something good.",
        "THE DOG DIED": "The dog died.",
    }
    bt = lambda text: canned.get(text, "")
    src = "The dog died."
    good = fitness(src, "BEFORE NOW, SOMETHING THAT LIVED DIED", bt)
    wrong = fitness(src, "BEFORE NOW, SOMEONE FELT SOMETHING GOOD", bt)
    copy = fitness(src, "THE DOG DIED", bt)
    junk = fitness(src, "[dog?] [died?]", bt)
    for label, r in (("valid + faithful", good), ("valid, wrong meaning", wrong),
                     ("copied the source", copy), ("unknown markers", junk)):
        print(f"    {label:22} fitness={r.fitness:.3f}  validity={r.validity:.2f}  "
              f"faithfulness={r.faithfulness:.2f}")
    expect("valid + faithful beats valid-but-wrong", good.fitness > wrong.fitness)
    expect("valid + faithful beats a verbatim copy", good.fitness > copy.fitness)
    expect("all-invalid candidate scores 0 and skips the round trip", junk.fitness == 0.0 and junk.back_translation == "")
    expect("fitness stays within 0..1", all(0.0 <= r.fitness <= 1.0 for r in (good, wrong, copy, junk)))

    print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED")
    return 0 if ok else 1


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Fitness function for English -> semantic primes.")
    ap.add_argument("--check", metavar="TEXT", help="validity check of prime text (offline)")
    ap.add_argument("--strict", action="store_true", help="no grammar words allowed")
    ap.add_argument("--selftest", action="store_true", help="run built-in tests")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    if args.check:
        v = check_validity(args.check, strict=args.strict)
        print(f"validity ratio : {v.ratio:.2f}  ({'ALL VALID' if v.all_valid else 'has invalid tokens'})")
        if v.invalid_tokens:
            print("invalid        :", ", ".join(v.invalid_tokens))
        if v.grammar_tokens:
            print("grammar (ok)   :", ", ".join(v.grammar_tokens))
        if v.names:
            print("names          :", ", ".join(v.names))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())