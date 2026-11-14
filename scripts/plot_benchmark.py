#!/usr/bin/env python3
"""Render the scalar/SSE2/AVX2 benchmark comparison as light and dark SVGs.

Reads one or more hft_bench CSV files, pools every sample row, and plots the
median phase totals. Only the standard library is required.

Usage:
    python3 scripts/plot_benchmark.py \
        --input results-1.csv results-2.csv results-3.csv --out-dir docs
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import statistics
from pathlib import Path

PHASES = (
    ("insert_ms", "Insert"),
    ("lookup_ms", "Lookup"),
    ("range_ms", "Range"),
    ("update_ms", "Update"),
)
SERIES = (
    ("scalar", "scalar", "series-1"),
    ("sse2", "SSE2", "series-2"),
    ("avx2", "AVX2", "series-3"),
)

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text-primary": "#0b0b0b",
        "text-secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "baseline": "#c3c2b7",
        "series-1": "#2a78d6",
        "series-2": "#eb6834",
        "series-3": "#1baf7a",
    },
    "dark": {
        "surface": "#1a1a19",
        "text-primary": "#f0efec",
        "text-secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "baseline": "#383835",
        "series-1": "#3987e5",
        "series-2": "#d95926",
        "series-3": "#199e70",
    },
}

WIDTH, HEIGHT = 960, 560
MARGIN_LEFT, MARGIN_RIGHT = 64, 32
MARGIN_TOP, MARGIN_BOTTOM = 120, 44
H_GAP, V_GAP = 36, 34
BAR_WIDTH, BAR_GAP = 20, 2
CORNER = 4


def read_medians(paths: list[Path]) -> dict[str, dict[str, float]]:
    samples: dict[str, dict[str, list[float]]] = {}
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("record") != "sample":
                    continue
                backend = row.get("backend")
                if backend not in {name for name, _, _ in SERIES}:
                    continue
                bucket = samples.setdefault(backend, {})
                for column, _ in PHASES:
                    value = row.get(column, "")
                    if value:
                        bucket.setdefault(column, []).append(float(value))
    return {
        backend: {
            column: statistics.median(values)
            for column, values in columns.items()
            if values
        }
        for backend, columns in samples.items()
    }


def nice_axis(maximum: float) -> tuple[float, float]:
    """Return (axis_max, tick_step) with four evenly spaced ticks."""
    target = max(maximum * 1.12, 1e-9)
    exponent = math.floor(math.log10(target))
    base = 10.0 ** exponent
    for multiplier in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        step = multiplier * base
        if step * 4 >= target:
            return step * 4, step
    return base * 40, base * 10


def fmt_tick(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:g}"


def rounded_bar(x: float, y: float, width: float, height: float) -> str:
    """A bar with a 4px rounded data-end and a square baseline."""
    radius = min(CORNER, height)
    bottom = y + height
    return (
        f"M{x:.2f},{bottom:.2f} "
        f"L{x:.2f},{y + radius:.2f} "
        f"Q{x:.2f},{y:.2f} {x + radius:.2f},{y:.2f} "
        f"L{x + width - radius:.2f},{y:.2f} "
        f"Q{x + width:.2f},{y:.2f} {x + width:.2f},{y + radius:.2f} "
        f"L{x + width:.2f},{bottom:.2f} Z"
    )


def render(theme_name: str, data: dict[str, dict[str, float]], path: Path) -> None:
    theme = THEMES[theme_name]
    present = [
        (name, label, theme[slot])
        for name, label, slot in SERIES
        if name in data and any(column in data[name] for column, _ in PHASES)
    ]
    if not present:
        raise SystemExit("no scalar/SSE2/AVX2 sample rows found")

    plot_width = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    plot_height = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    panel_width = (plot_width - H_GAP) / 2
    panel_height = (plot_height - V_GAP) / 2
    plot_inner_width = panel_width - 44
    plot_inner_height = panel_height - 26
    group_width = len(present) * BAR_WIDTH + (len(present) - 1) * BAR_GAP

    out: list[str] = []
    add = out.append
    add(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" role="img" '
        f'aria-labelledby="chart-title chart-desc" '
        f'font-family="system-ui, -apple-system, \'Segoe UI\', sans-serif">'
    )
    add('<title id="chart-title">SIMD backend comparison</title>')
    add(
        '<desc id="chart-desc">Four panels compare insert, lookup, range, and '
        "update phase totals for the scalar, SSE2, and AVX2 search backends. "
        "Lower is better; each panel has its own y-scale. Exact values are in "
        "the benchmark table in the README.</desc>"
    )
    add(f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{theme["surface"]}"/>')

    add(
        f'<text x="{MARGIN_LEFT}" y="34" font-size="18" font-weight="600" '
        f'fill="{theme["text-primary"]}">SIMD backend comparison</text>'
    )
    add(
        f'<text x="{MARGIN_LEFT}" y="58" font-size="12" '
        f'fill="{theme["text-secondary"]}">100k keys · 200k lookups · 5k ranges · '
        "20k updates · medians of 21 samples · each panel has its own scale · "
        "lower is better</text>"
    )

    legend_x = MARGIN_LEFT
    for _, label, color in present:
        add(
            f'<rect x="{legend_x}" y="78" width="10" height="10" rx="2" '
            f'fill="{color}"/>'
        )
        add(
            f'<text x="{legend_x + 16}" y="87" font-size="12" '
            f'fill="{theme["text-secondary"]}">{html.escape(label)}</text>'
        )
        legend_x += 100

    for index, (column, title) in enumerate(PHASES):
        col = index % 2
        row = index // 2
        panel_x = MARGIN_LEFT + col * (panel_width + H_GAP)
        panel_y = MARGIN_TOP + row * (panel_height + V_GAP)
        plot_left = panel_x + 44
        plot_top = panel_y + 26
        plot_bottom = panel_y + panel_height
        values = [
            data[name][column]
            for name, _, _ in present
            if column in data[name]
        ]
        if not values:
            continue
        axis_max, tick_step = nice_axis(max(values))

        add(
            f'<text x="{panel_x}" y="{panel_y + 13}" font-size="13" '
            f'font-weight="600" fill="{theme["text-primary"]}">'
            f"{html.escape(title)}</text>"
        )

        for tick_index in range(5):
            tick = tick_index * tick_step
            y = plot_bottom - (tick / axis_max) * plot_inner_height
            stroke = theme["baseline"] if tick_index == 0 else theme["grid"]
            add(
                f'<line x1="{plot_left}" y1="{y:.2f}" x2="{plot_left + plot_inner_width}" '
                f'y2="{y:.2f}" stroke="{stroke}" stroke-width="1"/>'
            )
            add(
                f'<text x="{plot_left - 8}" y="{y + 4:.2f}" font-size="10" '
                f'text-anchor="end" fill="{theme["muted"]}" '
                f'font-variant-numeric="tabular-nums">{fmt_tick(tick)}</text>'
            )

        group_x = plot_left + (plot_inner_width - group_width) / 2
        for series_index, (name, label, color) in enumerate(present):
            if column not in data[name]:
                continue
            value = data[name][column]
            height = (value / axis_max) * plot_inner_height
            x = group_x + series_index * (BAR_WIDTH + BAR_GAP)
            y = plot_bottom - height
            add(
                f'<g><title>{html.escape(label)}, {html.escape(title.lower())}: '
                f"{value:.2f} ms</title>"
            )
            add(f'<path d="{rounded_bar(x, y, BAR_WIDTH, height)}" fill="{color}"/>')
            add("</g>")
            if name == "avx2":
                add(
                    f'<text x="{x + BAR_WIDTH / 2:.2f}" y="{y - 6:.2f}" '
                    f'font-size="10" text-anchor="middle" '
                    f'fill="{theme["text-secondary"]}" '
                    f'font-variant-numeric="tabular-nums">{value:.2f}</text>'
                )

    add(
        f'<text x="{MARGIN_LEFT}" y="{HEIGHT - 16}" font-size="10" '
        f'fill="{theme["muted"]}">Intel Core i5-10210U · GCC 16.2.1 · '
        "generated by scripts/plot_benchmark.py</text>"
    )
    add("</svg>")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True,
                        help="one or more hft_bench CSV files")
    parser.add_argument("--out-dir", type=Path, default=Path("docs"),
                        help="directory for benchmark-light.svg and benchmark-dark.svg")
    args = parser.parse_args()

    data = read_medians(args.input)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for theme_name in THEMES:
        render(theme_name, data, args.out_dir / f"benchmark-{theme_name}.svg")
    print(f"wrote {args.out_dir / 'benchmark-light.svg'} and {args.out_dir / 'benchmark-dark.svg'}")


if __name__ == "__main__":
    main()
