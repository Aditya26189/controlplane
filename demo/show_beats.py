"""Render the demo's six beats from committed artifacts. Layout only.

Every number comes from :mod:`controlplane.report.beats`, which reads
``results/``. This file draws boxes.

Usage:
    python demo/show_beats.py              # all six
    python demo/show_beats.py --beat 2     # one
    python demo/show_beats.py --plain      # no colour, for recording
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controlplane.report.beats import Beat, assemble_beats  # noqa: E402

WIDTH = 100
BOLD, DIM, RED, GREEN, CYAN, RESET = (
    "\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[36m", "\033[0m",
)


def paint(text: str, code: str, enabled: bool) -> str:
    return f"{code}{text}{RESET}" if enabled else text


def wrap(text: str, width: int, indent: str = "") -> list[str]:
    words, lines, line = text.split(), [], indent
    for word in words:
        if len(line) + len(word) + 1 > width and line.strip():
            lines.append(line.rstrip())
            line = indent + word + " "
        else:
            line += word + " "
    if line.strip():
        lines.append(line.rstrip())
    return lines


def render(beat: Beat, colour: bool) -> str:
    out = [paint("=" * WIDTH, DIM, colour)]
    out.append(paint(f"  BEAT {beat.number} - {beat.title}", BOLD, colour))
    out.append(paint("=" * WIDTH, DIM, colour))
    out.append("")
    for line in wrap(f"answers: {beat.answers}", WIDTH - 4, "  "):
        out.append(paint(line, DIM, colour))
    out.append("")

    if beat.missing:
        out.append(paint(f"  BEAT UNAVAILABLE - {beat.missing}", RED, colour))
        out.append("")
        return "\n".join(out)

    label_width = max((len(label) for label, _ in beat.rows), default=0)
    for label, value in beat.rows:
        first = f"  {label:<{label_width}}  "
        wrapped = wrap(value, WIDTH - len(first), "")
        out.append(paint(first, CYAN, colour) + (wrapped[0] if wrapped else ""))
        for extra in wrapped[1:]:
            out.append(" " * len(first) + extra)

    if beat.note:
        out.append("")
        for line in wrap(beat.note, WIDTH - 4, "  "):
            out.append(paint(line, BOLD, colour))

    if beat.artifacts:
        out.append("")
        out.append(paint("  check it:", DIM, colour))
        for artifact in beat.artifacts:
            out.append(paint(f"    {artifact}", DIM, colour))
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beat", type=int, default=None, help="render one beat")
    parser.add_argument("--plain", action="store_true", help="no colour")
    args = parser.parse_args(argv)

    colour = not args.plain and sys.stdout.isatty()
    beats = assemble_beats(PROJECT_ROOT)
    if args.beat is not None:
        beats = tuple(b for b in beats if b.number == args.beat)
        if not beats:
            print(f"no beat {args.beat}; there are 1..6", file=sys.stderr)
            return 2

    print()
    for beat in beats:
        print(render(beat, colour))

    unavailable = [b for b in beats if b.missing]
    if unavailable:
        print(
            f"  {len(unavailable)} of {len(beats)} beats could not be assembled. "
            "Reported rather than skipped.",
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
