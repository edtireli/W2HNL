#!/usr/bin/env python3
"""Debug visualizations for extracting Fig. 7a/7b purple W→Nl marker points.

What it does:
- Renders each PDF page to an image.
- Finds the axes frame (largest black stroked rectangle).
- Finds the purple square markers (filled shapes with the known purple RGB).
- Overlays on the rendered image:
  - frame rectangle
  - marker centers
  - projection lines to the frame edges ("extrapolation" to axes)
  - tick-label bounding boxes for quick sanity (text extraction)
- Maps marker centers into data coordinates using the frame rectangle and
  assumed axis ranges/scales (documented in the figure itself).
- Saves:
  - tools/_digitize_debug/fig_07a_overlay.png
  - tools/_digitize_debug/fig_07a_mapped_points.png
  - tools/_digitize_debug/fig_07b_overlay.png
  - tools/_digitize_debug/fig_07b_mapped_points.png

Run:
  python tools/digitize_fig07_debug.py
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import numpy as np


PURPLE = (0.506, 0.149, 0.961)  # W→Nl series stroke/fill color in both PDFs
BLACK = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class AxisMapping:
    xscale: str  # 'linear' or 'log10'
    xmin: float
    xmax: float
    ymin: float = 0.0
    ymax: float = 1.8


def _round_col(c):
    if c is None:
        return None
    return tuple(round(float(x), 3) for x in c)


def _get_frame_rect(page: fitz.Page) -> fitz.Rect:
    """Largest black stroked rectangle after applying rotation transform."""
    rot = page.rotation_matrix
    best = None
    max_area = -1.0
    for d in page.get_drawings():
        if _round_col(d.get("color")) != BLACK:
            continue
        if d.get("type") != "s":
            continue
        r = fitz.Rect(d["rect"]) * rot
        area = float(r.width * r.height)
        if area > max_area:
            max_area = area
            best = r
    if best is None:
        raise RuntimeError("Could not find axes frame rectangle")
    return best


def _get_purple_marker_rects(page: fitz.Page) -> List[fitz.Rect]:
    """Filled purple rectangles (the square markers) transformed to rotated coords."""
    rot = page.rotation_matrix
    rects: List[fitz.Rect] = []
    for d in page.get_drawings():
        if d.get("type") != "f":
            continue
        if _round_col(d.get("fill")) != PURPLE:
            continue
        rects.append(fitz.Rect(d["rect"]) * rot)
    return rects


def _dedup_points(points: Iterable[Tuple[float, float]], tol: float = 1e-6):
    out: List[Tuple[float, float]] = []
    for x, y in sorted(points):
        if out and abs(x - out[-1][0]) < tol and abs(y - out[-1][1]) < tol:
            continue
        out.append((x, y))
    return out


def _map_to_data(
    frame: fitz.Rect,
    x_page: float,
    y_page: float,
    mapping: AxisMapping,
) -> Tuple[float, float]:
    """Map a (x,y) point from rotated page coords into data coords."""
    tx = (x_page - frame.x0) / (frame.x1 - frame.x0)
    tx = float(np.clip(tx, 0.0, 1.0))

    if mapping.xscale == "linear":
        x_data = mapping.xmin + tx * (mapping.xmax - mapping.xmin)
    elif mapping.xscale == "log10":
        x_data = 10 ** (
            math.log10(mapping.xmin)
            + tx * (math.log10(mapping.xmax) - math.log10(mapping.xmin))
        )
    else:
        raise ValueError(f"Unknown xscale: {mapping.xscale}")

    ty = (frame.y1 - y_page) / (frame.y1 - frame.y0)  # y increases upward in data
    ty = float(np.clip(ty, 0.0, 1.0))
    y_data = mapping.ymin + ty * (mapping.ymax - mapping.ymin)

    return float(x_data), float(y_data)


def _extract_tick_spans(page: fitz.Page):
    """Return text spans that look like tick labels (digits, decimals), in rotated coords."""
    rot = page.rotation_matrix
    spans = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for s in l.get("spans", []):
                t = (s.get("text") or "").strip()
                if not t:
                    continue
                if not re.fullmatch(r"[0-9]+(\.[0-9]+)?", t):
                    continue
                bb = fitz.Rect(s["bbox"]) * rot
                spans.append((t, bb))
    return spans


def _render_page_to_image(page: fitz.Page, dpi: int = 250) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def _page_to_pixel(page: fitz.Page, frame: fitz.Rect, x: float, y: float, img: np.ndarray):
    # after rotation_matrix we are in the same coordinate system as page.rect
    pr = page.rect
    w, h = img.shape[1], img.shape[0]
    px = (x - pr.x0) / (pr.x1 - pr.x0) * w
    py = (y - pr.y0) / (pr.y1 - pr.y0) * h
    return px, py


def debug_one(pdf_path: str, mapping: AxisMapping, outdir: str):
    doc = fitz.open(pdf_path)
    page = doc[0]

    frame = _get_frame_rect(page)
    marker_rects = _get_purple_marker_rects(page)
    marker_centers = [((r.x0 + r.x1) / 2.0, (r.y0 + r.y1) / 2.0) for r in marker_rects]

    # map to data
    mapped = _dedup_points([_map_to_data(frame, x, y, mapping) for x, y in marker_centers])

    # render image
    img = _render_page_to_image(page, dpi=250)

    # --- Overlay plot (image space) ---
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)
    ax.set_title(os.path.basename(pdf_path) + " — PDF render + overlays")
    ax.set_axis_off()

    # frame
    fx0, fy0 = _page_to_pixel(page, frame, frame.x0, frame.y0, img)
    fx1, fy1 = _page_to_pixel(page, frame, frame.x1, frame.y1, img)
    ax.add_patch(
        plt.Rectangle(
            (fx0, fy0),
            fx1 - fx0,
            fy1 - fy0,
            fill=False,
            linewidth=2.0,
            edgecolor="cyan",
            label="Detected axes frame",
        )
    )

    # markers + projection lines to axes
    for (x, y) in marker_centers:
        px, py = _page_to_pixel(page, frame, x, y, img)
        ax.plot(px, py, marker="o", markersize=4, color="magenta")

        # project to left and bottom edges of frame
        p_left, _ = _page_to_pixel(page, frame, frame.x0, y, img)
        _, p_bottom = _page_to_pixel(page, frame, x, frame.y1, img)
        ax.plot([p_left, px], [py, py], color="magenta", linewidth=0.6, alpha=0.7)
        ax.plot([px, px], [py, p_bottom], color="magenta", linewidth=0.6, alpha=0.7)

    # tick span boxes
    spans = _extract_tick_spans(page)
    for text, bb in spans:
        x0, y0 = _page_to_pixel(page, frame, bb.x0, bb.y0, img)
        x1, y1 = _page_to_pixel(page, frame, bb.x1, bb.y1, img)
        ax.add_patch(
            plt.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                linewidth=0.8,
                edgecolor="yellow",
                alpha=0.6,
            )
        )
        ax.text(x0, y0 - 2, text, color="yellow", fontsize=8)

    ax.legend(loc="lower right")

    overlay_path = os.path.join(outdir, os.path.splitext(os.path.basename(pdf_path))[0] + "_overlay.png")
    fig.tight_layout()
    fig.savefig(overlay_path, dpi=200)
    plt.close(fig)

    # --- Mapped points plot (data space) ---
    xs = np.array([p[0] for p in mapped], dtype=float)
    ys = np.array([p[1] for p in mapped], dtype=float)

    fig2 = plt.figure(figsize=(7.5, 5.5))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.scatter(xs, ys, s=35)
    ax2.set_xlabel("|d0| [mm]" if "07a" in pdf_path else "R_prod [mm]")
    ax2.set_ylabel("Track reconstruction efficiency")
    ax2.set_ylim(mapping.ymin, mapping.ymax)
    if mapping.xscale == "log10":
        ax2.set_xscale("log")
        ax2.set_xlim(mapping.xmin, mapping.xmax)
    else:
        ax2.set_xlim(mapping.xmin, mapping.xmax)

    # annotate point indices
    for i, (x, y) in enumerate(mapped, start=1):
        ax2.annotate(str(i), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)

    ax2.grid(True, alpha=0.3)
    ax2.set_title(os.path.basename(pdf_path) + " — extracted marker points (mapped)")

    mapped_path = os.path.join(outdir, os.path.splitext(os.path.basename(pdf_path))[0] + "_mapped_points.png")
    fig2.tight_layout()
    fig2.savefig(mapped_path, dpi=200)
    plt.close(fig2)

    # also dump the points as text
    txt_path = os.path.join(outdir, os.path.splitext(os.path.basename(pdf_path))[0] + "_mapped_points.tsv")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("# x\ty\n")
        for x, y in mapped:
            f.write(f"{x:.10g}\t{y:.10g}\n")

    return overlay_path, mapped_path, txt_path, mapped


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outdir = os.path.join(repo_root, "tools", "_digitize_debug")
    os.makedirs(outdir, exist_ok=True)

    # Axis ranges inferred from the figure ticks:
    # - Fig 7a: x is log scale with ticks at 1, 10, 10^2 and axis appears to run ~0.3 to 300 mm
    # - Fig 7b: x is linear with ticks 0..300 mm
    mapping_a = AxisMapping(xscale="log10", xmin=0.3, xmax=300.0)
    mapping_b = AxisMapping(xscale="linear", xmin=0.0, xmax=300.0)

    a = debug_one(os.path.join(repo_root, "fig_07a.pdf"), mapping_a, outdir)
    b = debug_one(os.path.join(repo_root, "fig_07b.pdf"), mapping_b, outdir)

    print("Wrote:")
    for p in [a[0], a[1], a[2], b[0], b[1], b[2]]:
        print(" -", os.path.relpath(p, repo_root))


if __name__ == "__main__":
    main()
