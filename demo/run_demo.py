"""Two-pane demo runner. Renders; decides nothing.

``CLAUDE.md``: no logic in the demo runner. Every number on screen comes from
:mod:`src.demo.session`, which uses the same probe, warrant, certificate and
ledger code as the pipeline. This file lays out text.

Usage:
    python demo/run_demo.py --fixture                 # play the stream, then prove it
    python demo/run_demo.py --fixture --events 8      # shorter
    python demo/run_demo.py --fixture --prove-only    # Beat 5 alone
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, set_seeds, setup_logging
from src.demo.session import DemoSession, RequestOutcome
from src.demo.stream import Stream, record_stream
from src.model import WarrantStatus
from src.store import Ledger
from src.validation.evalsets import SOURCE_SYNTHETIC
from src.validation.synthetic import synthetic_cache, synthetic_evalset

WIDTH = 62


def _use_utf8() -> bool:
    """Switch stdout to UTF-8 if it can be, and report whether glyphs are safe.

    A Windows console defaults to cp1252 and raises on box-drawing characters.
    Crashing a demo on a rendering detail in front of judges is the worst place
    to discover that, so the runner reconfigures where it can and falls back to
    ASCII where it cannot.
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, OSError, ValueError):
        pass
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "│─═·".encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


UTF8 = _use_utf8()
GAP = "  │  " if UTF8 else "  |  "
GLYPH = {"h": "─", "d": "═", "t": "·"} if UTF8 else {"h": "-", "d": "=", "t": "."}

BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RESET = "\033[0m"


def _supports_colour() -> bool:
    """Colour only when a terminal is attached and not explicitly disabled."""
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


COLOUR = _supports_colour()


def paint(text: str, code: str) -> str:
    return f"{code}{text}{RESET}" if COLOUR else text


def wrap(text: str, width: int = WIDTH) -> list[str]:
    """Hard-wrap to the pane width, preserving word boundaries where possible."""
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word[:width] if len(word) > width else word
    if current:
        lines.append(current)
    return lines or [""]


def two_columns(left: list[str], right: list[str]) -> str:
    """Lay two panes side by side, padding the shorter."""
    height = max(len(left), len(right))
    rows = []
    for i in range(height):
        a = left[i] if i < len(left) else ""
        b = right[i] if i < len(right) else ""
        rows.append(f"{a:<{WIDTH}}{GAP}{b}")
    return "\n".join(rows)


def rule(kind: str = "h") -> str:
    """A horizontal rule in whichever glyph set the console can render."""
    return GLYPH[kind] * (WIDTH * 2 + len(GAP))


def header(session: DemoSession, stream: Stream) -> str:
    banner = session.warrant_banner()
    synthetic = session.run.data_source == SOURCE_SYNTHETIC
    lines = [rule("d")]
    if synthetic:
        lines.append(
            paint(
                "  SYNTHETIC FIXTURE RUN — these numbers exercise the harness and "
                "are not measurements",
                RED + BOLD,
            )
        )
        lines.append(rule("d"))
    lines += [
        f"  {paint('CONVENTIONAL STACK', BOLD):<{WIDTH + (len(BOLD) + len(RESET) if COLOUR else 0)}}"
        f"{GAP}{paint('CONTROLPLANE', BOLD)}",
        rule(),
        "",
        paint("  WARRANT", CYAN + BOLD),
        f"    detector        {banner['detector']}",
        f"    operating point {banner['operating_point']} "
        f"(threshold {banner['threshold']:.4f}, selected on {banner['selected_on']})",
        f"    envelope        {banner['envelope']}",
        f"    status          {paint(banner['status'], GREEN if banner['status'] == 'VALID' else RED)}"
        f"  - validated {banner['validated']} · expires in {banner['expires_in_hours']}h",
        f"    recall          {banner['recall']}",
        f"    precision       {banner['precision']}",
        f"    AUROC           {banner['auroc']}   base rate {banner['base_rate']:.4f}",
        f"    flag rate       {banner['flag_rate']}",
        f"    yield           {banner['confirmed_errors']}",
        f"    kappa           {banner['kappa']}",
        "",
        f"  stream {stream.stream_id} - {len(stream)} requests - "
        f"from {stream.source_eval_set}",
        rule(),
    ]
    return "\n".join(lines)


def render_outcome(outcome: RequestOutcome, index: int, total: int) -> str:
    """One request, both panes."""
    event = outcome.event
    left, right = outcome.left, outcome.right

    flag_colour = YELLOW if left.flagged else GREEN
    left_lines = [
        paint(f"  {left.headline}", flag_colour + BOLD),
        f"  score {left.score:.4f}",
        "",
    ]
    for note in left.notes:
        left_lines += [paint(f"  {line}", DIM) for line in wrap(note, WIDTH - 2)]

    right_lines = [
        paint(f"  {right.headline}", flag_colour + BOLD),
        f"  score {right.score:.4f}",
        "",
    ]
    if right.bounds:
        recall = right.bounds["recall"]
        precision = right.bounds["precision"]
        right_lines += [
            paint("  claimed bounds", CYAN),
            f"    recall    {recall['value']:.3f} "
            f"[{recall['ci_low']:.3f}, {recall['ci_high']:.3f}] n={recall['n']}",
            f"    precision {precision['value']:.3f} "
            f"[{precision['ci_low']:.3f}, {precision['ci_high']:.3f}] n={precision['n']}",
        ]
    else:
        right_lines.append(paint("  no valid warrant — no bounds claimed", RED))
    if right.envelope:
        right_lines += [paint(f"  envelope  {right.envelope}", DIM)]
    if right.warrant_age:
        right_lines.append(paint(f"  warrant   validated {right.warrant_age} ago", DIM))
    for note in right.notes:
        right_lines += [paint(f"  {line}", DIM) for line in wrap(note, WIDTH - 2)]

    head = (
        f"[{index}/{total}] {event.request_id} - session {event.session_id} - "
        f"{event.token_length} tokens"
    )
    return "\n".join([paint(head, DIM), two_columns(left_lines, right_lines), rule("t")])


def render_proof(run, elapsed: float) -> str:
    """Beat 5: the five controls, including the fault being rejected."""
    lines = [
        rule("d"),
        paint("  PROVE IT — live validation", BOLD + CYAN),
        rule("d"),
        f"  {run.detector_id} / {run.variant} on {run.eval_set_id} [{run.envelope_id}]",
        f"  n_test {run.warrant.n_test} - base rate {run.base_rate:.4f} - "
        f"C {run.probe_fit.C:g} selected on {run.probe_fit.selected_on} · "
        f"completed in {elapsed:.2f}s",
        "",
        paint("  METRICS", BOLD),
    ]
    for metric in run.metrics.all_metrics():
        lines.append(f"    {metric.name:<20} {metric.render(4)}")
    lines += ["", paint("  CONTROLS", BOLD)]
    for control in run.controls:
        mark = paint("PASS", GREEN) if control.passed else paint("FAIL", RED)
        lines.append(
            f"    {mark}  {control.control:<15} measured {control.measured:.4f}  "
            f"margin {control.margin:+.4f}"
        )
        for line in wrap(control.detail, WIDTH * 2 - 8):
            lines.append(paint(f"          {line}", DIM))
    lines += [""]
    if run.warrant.status is WarrantStatus.VALID:
        lines.append(paint("  WARRANT ISSUED", GREEN + BOLD))
    else:
        lines.append(paint(f"  WARRANT {run.warrant.status.value}", RED + BOLD))
        for line in wrap(run.warrant.status_reason or "", WIDTH * 2 - 4):
            lines.append(paint(f"    {line}", RED))
    lines.append(rule("d"))
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument(
        "--fixture", action="store_true",
        help="run on the synthetic fixture; required until a real extraction exists",
    )
    parser.add_argument("--events", type=int, default=6, help="requests to play")
    parser.add_argument(
        "--delay", type=float, default=0.0, help="seconds between requests"
    )
    parser.add_argument(
        "--prove-only", action="store_true", help="skip the stream, run Beat 5 alone"
    )
    parser.add_argument(
        "--auto-prove", action="store_true",
        help="press Prove it automatically instead of waiting for a keypress",
    )
    parser.add_argument(
        "--variant", default="T1-max_rolling_means", help="tier variant to demo"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(level=40)  # errors only; the demo is the output
    config = load_config(args.config)
    set_seeds(config.seed)

    if not args.fixture:
        print(
            "Refusing to run without --fixture. A real extraction does not exist "
            "yet, and there is deliberately no default that would quietly "
            "substitute synthetic data for measured data (DECISIONS.md 027).",
            file=sys.stderr,
        )
        return 2

    evalset = synthetic_evalset(
        eval_set_id="triviaqa-600-synthetic", n_items=2400, base_rate=0.152,
        seed=config.seed, items_per_question=2, declare_splits=True,
    )
    cache = synthetic_cache(
        evalset, seed=config.seed,
        window=config.probe.rolling_window, stride=config.probe.rolling_stride,
    )
    canary_evalset = synthetic_evalset(
        eval_set_id="canary-20-synthetic", n_items=20, base_rate=0.95, seed=7
    )
    canary_cache = synthetic_cache(
        canary_evalset, seed=7,
        window=config.probe.rolling_window, stride=config.probe.rolling_stride,
        signal_by_tier={v: 8.0 for v in cache.variants}, amplitude_spread=0.05,
    )

    demo_db = PROJECT_ROOT / config.paths.results_dir / "demo.db"
    if demo_db.exists():
        demo_db.unlink()  # a demo run starts a fresh chain, so the trace is legible
    ledger = Ledger(demo_db, retention_days=config.store.retention_days)

    try:
        session = DemoSession(
            config, evalset, cache,
            variant=args.variant,
            detector_id=f"probe-{config.model.name.split('/')[-1].lower()}-{args.variant}",
            detector_version="0.1.0+fixture",
            ledger=ledger,
            canary_cache=canary_cache,
        )
        print(paint("preparing - validating once so the banner has something to show...", DIM))
        session.prepare()

        if not args.prove_only:
            stream = record_stream(
                evalset, cache, n_events=args.events, seed=config.seed
            )
            stream_path = PROJECT_ROOT / config.paths.results_dir / "demo_stream.json"
            stream.save(stream_path)

            print(header(session, stream))
            for position, event in enumerate(stream, start=1):
                print(render_outcome(session.handle(event), position, len(stream)))
                if args.delay:
                    time.sleep(args.delay)

            verification = ledger.verify_chain()
            print(
                f"  ledger: {verification.n_records} records, chain "
                f"{paint('intact', GREEN) if verification.ok else paint('BROKEN', RED)}"
            )
            print()

        if not args.auto_prove and sys.stdin.isatty():
            input(paint("  press Enter to run Prove it (live validation) ", BOLD))
        clock = time.perf_counter()
        run = session.prove_it()
        print(render_proof(run, time.perf_counter() - clock))
    finally:
        ledger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
