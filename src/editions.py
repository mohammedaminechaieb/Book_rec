"""
==============================================================================
File      : src/editions.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Ella (lead)  |  Amine (co-owner / reviewer)
------------------------------------------------------------------------------
What it does:
  Goodreads lists every edition of a book separately ("Batman, Vol. 1: The
  Court of Owls" and "Batman, Volume 1: The Court of Owls"; "Maus I ..." and
  "Maus: A Survivor's Tale ... (Maus, #1)"). Without care, a reader who picked
  one edition gets the other one recommended as a "new" book.
  This module gives each title a "work key" so editions of the same work can be
  recognised and filtered:
    - same series and number in parentheses, e.g. "(Maus, #1)"        -> same work
    - same normalised title (lowercase, no "vol."/"volume", no "#n",
      no punctuation, no leading "the")                                 -> same work
    - collected editions ("The Complete Maus", "... Omnibus") of a series
      the reader already picked                                         -> same work
Input:
  - book titles (str)
Output:
  - work keys, and a filter that drops candidates that repeat a picked/seen work
==============================================================================
"""
import re

_PARENS = re.compile(r"\(([^()]*)\)")
_SERIES = re.compile(r"^\s*(.*?),?\s*#\s*([\d.]+)\s*$")
_COLLECTED = re.compile(r"\b(complete|omnibus|collection|collected|intégrale|integrale|library|box set)\b")


def _norm(text: str) -> str:
    text = str(text).lower()
    text = text.replace("volume", "vol")
    text = re.sub(r"\bvol\.?\s*[\divxlc]+\b", " ", text)      # "vol. 1", "vol iii"
    text = re.sub(r"#\s*[\d.]+", " ", text)                     # "#1"
    text = re.sub(r"[^\w\s]", " ", text)                         # punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"^the ", "", text)


def series_key(title: str) -> str | None:
    """'Maus I (Maus, #1)' -> 'maus#1'; None if the title has no '(Series, #n)'."""
    for inside in _PARENS.findall(str(title)):
        m = _SERIES.match(inside)
        if m and m.group(1).strip():
            return f"{_norm(m.group(1))}#{m.group(2)}"
    return None


def series_name(title: str) -> str | None:
    key = series_key(title)
    return key.split("#")[0] if key else None


def title_key(title: str) -> str:
    """Normalised title without the '(Series, #n)' part."""
    return _norm(_PARENS.sub(" ", str(title)))


def work_keys(title: str) -> set[str]:
    """All keys that identify the work of this title (used for matching)."""
    keys = {"t:" + title_key(title)}
    s = series_key(title)
    if s:
        keys.add("s:" + s)
    return keys


def is_same_work(candidate: str, reference_keys: set[str], reference_series: set[str]) -> bool:
    """True if candidate is an edition (or a collected edition) of a reference work."""
    if work_keys(candidate) & reference_keys:
        return True
    norm = title_key(candidate)
    if _COLLECTED.search(norm):
        return any(name and re.search(rf"\b{re.escape(name)}\b", norm) for name in reference_series)
    return False


def dedupe_candidates(candidate_titles: list[str], reference_titles: list[str], limit: int) -> list[int]:
    """Positions (in candidate order) of the first `limit` candidates that are new works.

    A candidate is dropped if it is an edition of a reference work (picked or
    already read) or of a candidate kept earlier in the list.
    """
    ref_keys: set[str] = set()
    ref_series: set[str] = set()
    for t in reference_titles:
        ref_keys |= work_keys(t)
        name = series_name(t)
        # series name, or the part of the title before ':' / ',' ("Persepolis: ..." -> "persepolis")
        ref_series.add(name or _norm(re.split(r"[:,(]", str(t))[0]))
    kept: list[int] = []
    for pos, title in enumerate(candidate_titles):
        if is_same_work(title, ref_keys, ref_series):
            continue
        kept.append(pos)
        ref_keys |= work_keys(title)
        if len(kept) >= limit:
            break
    return kept
