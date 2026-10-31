"""Advice shared by all sports: short tips in plain language, each with a label.

Every tip has one of three kinds, so that the reader knows how solid it is:

  RESEARCH  measured in the video AND supported by a study
  COACH     measured in the video, but it is common coaching advice that the
            studies we read did not test
  CHECK     something the camera cannot see: the person has to check it alone

The report shows at most MAX_MAIN measured tips first (the most important ones),
then the "check it yourself" list. Everything else goes to the details.
"""

from dataclasses import dataclass

RESEARCH = "research"
COACH = "coach"
CHECK = "check"

MAX_MAIN = 3   # assumption: more than three things to fix at once is too much

# Words shown to the reader (plain language, no jargon)
KIND_LABELS = {
    RESEARCH: "backed by research",
    COACH: "coach's tip",
    CHECK: "check it yourself",
}

SURE_WORDS = {
    "high": "fairly sure",
    "medium": "moderately sure",
    "low": "not very sure",
}


@dataclass
class Advice:
    """One tip. `saw` is empty for CHECK tips (nothing was measured)."""

    kind: str
    title: str          # short and plain, e.g. "Your heels come off the floor"
    try_this: str       # what to do, in one or two sentences
    why: str            # the reason, in one plain sentence
    saw: str = ""       # what the video showed
    sure: str = "medium"  # "high", "medium" or "low": how sure the measurement and rule are
    priority: int = 5   # lower number = shown first
    source: str = ""    # short reference, shown only in the details


def advice_lines(item):
    """Format one tip as report lines: a bullet, then indented notes (shown small in the app)."""
    # The "check it yourself" tips already sit under their own heading: no label needed
    label = "" if item.kind == CHECK else f" ({KIND_LABELS[item.kind]})"
    lines = [f"- {item.title}{label}"]
    if item.kind == CHECK:
        lines.append(f"  What to do: {item.try_this}")
        lines.append(f"  Why: {item.why}")
    else:
        if item.saw:
            lines.append(f"  What I saw: {item.saw}")
        lines.append(f"  What to try: {item.try_this}")
        lines.append(f"  Why: {item.why}")
        lines.append(f"  How sure I am: {SURE_WORDS[item.sure]}.")
    return lines


def legend_lines():
    """Explain the three labels in one short note."""
    return [
        "  Labels: 'backed by research' = I measured it in your video and a study supports it. "
        "'coach's tip' = I measured it, but it is common coaching advice that the studies I read "
        "did not test. 'check it yourself' = the camera cannot see it.",
    ]


def split_items(items, max_main=MAX_MAIN):
    """Return (main, extra, checks): measured tips sorted by importance, and the CHECK tips."""
    measured = sorted((a for a in items if a.kind != CHECK),
                      key=lambda a: (a.priority, a.kind != RESEARCH))
    checks = [a for a in items if a.kind == CHECK]
    return measured[:max_main], measured[max_main:], checks


def what_to_do_section(items, nothing_to_fix, max_main=MAX_MAIN):
    """The first part of the report: what to change, most important first."""
    main, _, _ = split_items(items, max_main)
    lines = []
    if main:
        for item in main:
            lines += advice_lines(item)
    else:
        lines.append(f"- {nothing_to_fix}")
    lines += legend_lines()
    return ("WHAT TO DO", lines)


def check_section(items):
    """The things the camera cannot see, shown to everyone."""
    _, _, checks = split_items(items)
    lines = []
    for item in checks:
        lines += advice_lines(item)
    return ("CHECK IT YOURSELF", lines)


def extra_section(items, max_main=MAX_MAIN):
    """Measured tips that did not fit in the main list (for the details), or None."""
    _, extra, _ = split_items(items, max_main)
    if not extra:
        return None
    lines = []
    for item in extra:
        lines += advice_lines(item)
    return ("OTHER POINTS", lines)


def source_lines(items):
    """One line per tip with its reference (for the details)."""
    lines = []
    for item in items:
        if item.source:
            lines.append(f"- {item.title}: {item.source}")
    return lines