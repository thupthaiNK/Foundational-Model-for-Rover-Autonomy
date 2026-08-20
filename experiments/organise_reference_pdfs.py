"""
Purpose: Sort the reference PDF library into two delivery folders: the papers
         actually cited in the thesis, numbered in the order they first appear
         in the report, and everything else, numbered alphabetically. Copies
         only; the originals in Ref/ are left exactly where they are.
Inputs:  docs/word_transfer/references.md  (the APA reference list)
         docs/word_transfer/ch*.md, appendix_*.md  (to find first appearance)
         C:/Users/DELL/Desktop/Thesis/Ref/**/*.pdf  (the source library)
Outputs: C:/Users/DELL/Desktop/Thesis/Ref/Ref Report/NN. Author Year - Title.pdf
         C:/Users/DELL/Desktop/Thesis/Ref/Ref Not Cited/NNN. Title.pdf
         and a report on stdout listing any citation with no matching PDF.
How to run: python3 experiments/organise_reference_pdfs.py [--apply]
         Without --apply it only prints the plan and changes nothing.
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import os
import re
import shutil
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.join(HERE, "..", "docs", "word_transfer")
REF_ROOT = "/mnt/c/Users/DELL/Desktop/Thesis/Ref"
CITED_DIR = os.path.join(REF_ROOT, "Ref Report")
UNCITED_DIR = os.path.join(REF_ROOT, "Ref Not Cited")

# The order a reader meets the text. Front matter and the reference list
# itself are excluded: a citation's "first appearance" means its first use in
# an argument, not its entry in the alphabetical list at the back.
READING_ORDER = [
    "ch1_introduction.md", "ch2_literature_review.md", "ch3_methodology.md",
    "ch4_results.md", "ch5_discussion.md", "ch6_conclusion.md",
    "appendices_intro.md", "appendix_a_code.md", "appendix_b_reproducibility.md",
    "appendix_c_full_model_table.md", "appendix_d_negative_results.md",
]

# Words that carry no matching signal. Dropped before comparing a reference
# title against a filename, so that "A Survey of X" and "X: A Survey" still
# match each other.
STOPWORDS = {
    "a", "an", "and", "the", "of", "for", "in", "on", "to", "with", "from",
    "at", "by", "as", "is", "are", "its", "their", "this", "that", "or",
}


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def tokens(s):
    """Comparable word set: accents folded, punctuation dropped, stopwords
    removed. Digits are kept, since version numbers (v2, v3, SAM2) are often
    the only thing separating two otherwise identical titles."""
    s = strip_accents(s).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return {w for w in s.split() if w not in STOPWORDS and len(w) > 1}


def parse_references(path):
    """Split the APA list into (authors, year, title) per entry.

    An entry starts at a line beginning with a capitalised surname followed by
    a comma, which is what distinguishes it from a continuation line."""
    with open(path) as f:
        text = f.read()
    entries = []
    for block in text.split("\n\n"):
        block = " ".join(block.split())
        if not block or block.startswith("#"):
            continue
        m = re.match(r"^(.*?)\s*\((\d{4}[a-z]?)\)\.\s*(.*)$", block)
        if not m:
            continue
        authors, year, rest = m.groups()
        # Title runs to the first full stop that is not inside an initial or
        # an abbreviation. Splitting on ". *" (the italic journal marker) or
        # on a sentence-ending full stop covers every entry in this list.
        title = re.split(r"\.\s*\*|\.\s*https?://|\.\s*$|\.\s+[A-Z]", rest)[0]
        entries.append({
            "authors": authors,
            "year": year[:4],
            "title": title.strip().rstrip("."),
            "raw": block,
        })
    return entries


def surname(authors):
    """First author's surname, for building the citation key and the filename."""
    first = authors.split(",")[0].strip()
    return strip_accents(first).replace(" ", "")


def find_first_appearance(entries):
    """Position of each reference's first in-text citation, as a
    (file index, character offset) pair. Uncited entries get None."""
    chapters = []
    for name in READING_ORDER:
        p = os.path.join(WT, name)
        if os.path.exists(p):
            with open(p) as f:
                chapters.append(strip_accents(f.read()))
        else:
            chapters.append("")

    for e in entries:
        sn = surname(e["authors"])
        # Matches "Smith (2020)", "Smith et al. (2020)", "(Smith et al., 2020)"
        # and the two- and three-author "Smith & Jones (2020)" forms.
        pat = re.compile(re.escape(sn) + r"[^)]{0,80}?" + e["year"])
        e["first"] = None
        for idx, body in enumerate(chapters):
            m = pat.search(body)
            if m:
                e["first"] = (idx, m.start())
                break
    return entries


def collect_pdfs(root):
    out = []
    for dirpath, _dirnames, filenames in os.walk(root):
        # Never read back from the two folders this script writes.
        if os.path.abspath(dirpath).startswith(os.path.abspath(CITED_DIR)):
            continue
        if os.path.abspath(dirpath).startswith(os.path.abspath(UNCITED_DIR)):
            continue
        for fn in filenames:
            if fn.lower().endswith(".pdf"):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def clean_stem(path):
    """Filename without the extension and without any leading "12. " index
    the library already carries, which would otherwise pollute the match."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"^\s*\d+[.\-)]?\s*", "", stem)


def match(entry, pdfs):
    """Best PDF for a reference, by title-token overlap.

    Scored as the fraction of the reference title's words that appear in the
    filename. A filename is usually a truncated title, so recall against the
    reference title is the right direction to measure; requiring the reverse
    would reject every abbreviated filename in the library."""
    want = tokens(entry["title"])
    if not want:
        return None, 0.0
    best, best_score = None, 0.0
    for p in pdfs:
        have = tokens(clean_stem(p))
        score = len(want & have) / len(want)
        if score > best_score:
            best, best_score = p, score
    return best, best_score


def safe(name, limit=110):
    """Windows-safe filename, trimmed at a word boundary."""
    name = re.sub(r'[<>:"/\\|?*]', "", name)
    name = " ".join(name.split())
    if len(name) > limit:
        name = name[:limit].rsplit(" ", 1)[0]
    return name.rstrip(" .")


def main():
    apply = "--apply" in sys.argv
    entries = find_first_appearance(parse_references(os.path.join(WT, "references.md")))
    pdfs = collect_pdfs(REF_ROOT)

    cited = [e for e in entries if e["first"] is not None]
    uncited_refs = [e for e in entries if e["first"] is None]
    cited.sort(key=lambda e: e["first"])

    print(f"references parsed : {len(entries)}")
    print(f"cited in the text : {len(cited)}")
    print(f"never cited       : {len(uncited_refs)}")
    print(f"PDFs in library   : {len(pdfs)}\n")

    THRESHOLD = 0.5
    claimed, plan_cited, missing = set(), [], []
    for n, e in enumerate(cited, 1):
        pool = [p for p in pdfs if p not in claimed]
        p, score = match(e, pool)
        if p is None or score < THRESHOLD:
            missing.append((n, e, score))
            continue
        claimed.add(p)
        chap = READING_ORDER[e["first"][0]].split("_")[0].replace("ch", "Ch")
        dest = f"{n:02d}. [{chap}] {safe(surname(e['authors']) + ' ' + e['year'] + ' - ' + e['title'])}.pdf"
        plan_cited.append((p, dest, score))

    leftover = sorted([p for p in pdfs if p not in claimed], key=lambda p: clean_stem(p).lower())
    plan_uncited = [(p, f"{n:03d}. {safe(clean_stem(p))}.pdf")
                    for n, p in enumerate(leftover, 1)]

    print(f"=== CITED -> {CITED_DIR}  ({len(plan_cited)} matched) ===")
    for src, dest, score in plan_cited:
        print(f"  {dest}   [{score:.2f}]")

    if missing:
        print(f"\n=== NO PDF FOUND ({len(missing)}) -- these need finding by hand ===")
        for n, e, score in missing:
            print(f"  {n:02d}. {surname(e['authors'])} {e['year']} - {e['title'][:70]}  (best {score:.2f})")

    if uncited_refs:
        print(f"\n=== IN THE REFERENCE LIST BUT NEVER CITED ({len(uncited_refs)}) ===")
        for e in uncited_refs:
            print(f"  {surname(e['authors'])} {e['year']} - {e['title'][:70]}")

    print(f"\n=== NOT CITED -> {UNCITED_DIR}  ({len(plan_uncited)} files) ===")

    if not apply:
        print("\nDry run. Re-run with --apply to copy.")
        return

    for folder in (CITED_DIR, UNCITED_DIR):
        os.makedirs(folder, exist_ok=True)
    for src, dest, _ in plan_cited:
        shutil.copy2(src, os.path.join(CITED_DIR, dest))
    for src, dest in plan_uncited:
        shutil.copy2(src, os.path.join(UNCITED_DIR, dest))
    print(f"\ncopied {len(plan_cited)} cited and {len(plan_uncited)} uncited PDFs")


if __name__ == "__main__":
    main()
