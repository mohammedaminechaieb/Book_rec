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
    - same normalised title (lowercase, accents removed, "Vol." = "Volume",
      no "#n", no punctuation, no leading "the")                        -> same work
    - same series and volume written differently ("Persepolis, Volume 2" and
      "Persepolis 2: The Story of a Return")                            -> same work
    - same Goodreads work_id, when the data has it                      -> same work
    - a longer title starting with a picked title ("Persepolis: The Story of a
      Childhood and the Story of a Return")                             -> same work
    - collected editions ("The Complete Maus", "(Persepolis #1-4)", "Omnibus")
      of a series already picked, and single volumes of a series already
      covered by a collected edition in the list                        -> same work
Input:
  - book titles (str)
Output:
  - work keys, and a filter that drops candidates that repeat a picked/seen work
==============================================================================
"""
import re
import unicodedata

_PARENS = re.compile(r"\(([^()]*)\)")
_SERIES = re.compile(r"^\s*(.*?),?\s*#\s*([\d.]+)\s*$")
_RANGE = re.compile(r"^\s*(.*?),?\s*#\s*\d+\s*-\s*\d+\s*$")          # "(Persepolis #1-4)"
_COLLECTED = re.compile(r"\b(complete|omnibus|collection|collected|integrale|library|box(ed)? set|"
                        r"edicion integral|the story of a childhood and the story of a return)\b")
_VOLUME = [
    re.compile(r"^(.+?),?\s+(?:vol|volume|tome|book|tomo)\.?\s*(\d+)\b"),   # "Persepolis, Volume 2"
    re.compile(r"^([^:]+?)\s+(\d+)\s*:"),                                    # "Persepolis 2: ..."
]


def _ascii(text: str) -> str:
    """'Persépolis' -> 'Persepolis' (accents removed)."""
    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")


def _norm(text: str) -> str:
    text = _ascii(text).lower()
    text = text.replace("volume", "vol")
    text = re.sub(r"\bvol\.?\s*(\d+)\b", r" vol \1 ", text)    # "Vol. 1" / "Volume 1" -> "vol 1" (number kept)
    text = re.sub(r"#\s*[\d.]+", " ", text)                     # "#1"
    text = re.sub(r"[^\w\s]", " ", text)                         # punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"^the ", "", text)


def series_key(title: str) -> str | None:
    """'Maus I (Maus, #1)' -> 'maus#1', 'Persepolis, Volume 2' -> 'persepolis#2'; else None."""
    for inside in _PARENS.findall(_ascii(title)):
        m = _SERIES.match(inside)
        if m and m.group(1).strip():
            return f"{_norm(m.group(1))}#{float(m.group(2)):g}"
    bare = _PARENS.sub(" ", _ascii(title)).strip()
    for pattern in _VOLUME:
        m = pattern.match(bare.lower())
        if m and m.group(1).strip():
            return f"{_norm(m.group(1))}#{float(m.group(2)):g}"
    return None


def collected_series(title: str) -> str | None:
    """Series name of a collected edition ("(Persepolis #1-4)", "The Complete Maus"), else None."""
    for inside in _PARENS.findall(_ascii(title)):
        m = _RANGE.match(inside)
        if m and m.group(1).strip():
            return _norm(m.group(1))
    return None


def series_name(title: str) -> str | None:
    key = series_key(title)
    return key.split("#")[0] if key else None


def title_key(title: str) -> str:
    """Normalised title without the '(Series, #n)' part."""
    return _norm(_PARENS.sub(" ", str(title)))


def work_keys(title: str, work_id=None) -> set[str]:
    """All keys that identify the work of this title (used for matching).

    work_id: Goodreads work id when the data has it (same id = same work, any edition).
    """
    keys = {"t:" + title_key(title)}
    s = series_key(title)
    if s:
        keys.add("s:" + s)
    if work_id is not None and str(work_id) not in ("", "nan", "None"):
        keys.add("w:" + str(work_id))
    return keys


def _stem(title: str) -> str:
    """Series name, or the part of the title before ':' / ',' ("Persepolis: ..." -> "persepolis")."""
    s = series_key(title)
    if s:
        return s.split("#")[0]
    return _norm(re.split(r"[:,(]", _ascii(title))[0])


def is_same_work(candidate: str, reference_keys: set[str], reference_series: set[str],
                 reference_titles: set[str] | None = None, work_id=None,
                 collected_refs: set[str] | None = None) -> bool:
    """True if candidate is an edition (or a collected edition) of a reference work.

    collected_refs: series already covered by a collected edition in the references;
    a single volume of such a series is then a duplicate too.
    """
    if work_keys(candidate, work_id) & reference_keys:
        return True
    own = series_key(candidate)
    if collected_refs and own and own.split("#")[0] in collected_refs:
        return True
    norm = title_key(candidate)
    # a longer title that starts with a reference title ("... a Childhood and the Story of a Return")
    if reference_titles and any(len(r) >= 8 and norm.startswith(r + " ") for r in reference_titles):
        return True
    collected = collected_series(candidate)
    if collected and collected in reference_series:
        return True
    if _COLLECTED.search(norm):
        return any(name and re.search(rf"\b{re.escape(name)}\b", norm) for name in reference_series)
    return False


def dedupe_candidates(candidate_titles: list[str], reference_titles: list[str], limit: int,
                      candidate_work_ids: list | None = None,
                      reference_work_ids: list | None = None) -> list[int]:
    """Positions (in candidate order) of the first `limit` candidates that are new works.

    A candidate is dropped if it is an edition of a reference work (picked or
    already read) or of a candidate kept earlier in the list.
    """
    ref_keys: set[str] = set()
    ref_series: set[str] = set()
    ref_titles: set[str] = set()
    collected: set[str] = set()

    def remember(title, wid=None):
        ref_keys.update(work_keys(title, wid))
        ref_series.add(_stem(title))
        ref_titles.add(title_key(title))
        c = collected_series(title)
        if c:
            collected.add(c)

    for i, t in enumerate(reference_titles):
        remember(t, reference_work_ids[i] if reference_work_ids is not None else None)
    kept: list[int] = []
    for pos, title in enumerate(candidate_titles):
        wid = candidate_work_ids[pos] if candidate_work_ids is not None else None
        if is_same_work(title, ref_keys, ref_series, ref_titles, wid, collected):
            continue
        kept.append(pos)
        remember(title, wid)
        if len(kept) >= limit:
            break
    return kept
