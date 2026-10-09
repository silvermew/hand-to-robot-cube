"""Printable A4 sheets at exact physical size: table + wrist markers, cube markers, checkerboard.

    python -m src.make_markers      -> outputs/print/*.pdf

Print at 100% / "Actual size", never "Fit to page", then check the 100 mm scale bar
with a ruler. If poppler's pdftoppm is installed, the PDFs are rasterised and
the markers re-detected to check their printed size.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath

from src import constants as C

A4 = (210.0, 297.0)  # mm; all drawing below is in mm from the bottom-left corner
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "print"
DICTIONARY = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, C.ARUCO_DICT))
TABLE_DICTIONARY = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, C.TABLE_DICT))
GREY = "0.6"


def new_page(title):
    fig = plt.figure(figsize=(A4[0] / 25.4, A4[1] / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, A4[0])
    ax.set_ylim(0, A4[1])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(A4[0] / 2, A4[1] - 12, title, ha="center", va="top", fontsize=13, weight="bold")
    return fig, ax


def fill_cells(ax, cells, x0, y0, cell):
    """Fill grid cells (col, row-from-bottom) as one compound path so adjacent cells show no seams."""
    verts, codes = [], []
    for col, row in cells:
        x, y = x0 + col * cell, y0 + row * cell
        verts += [(x, y), (x + cell, y), (x + cell, y + cell), (x, y + cell), (x, y)]
        codes += [MplPath.MOVETO] + [MplPath.LINETO] * 3 + [MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor="black", edgecolor="none", lw=0))


def draw_marker(ax, marker_id, cx, cy, size_mm, quarter_turns=0, dictionary=DICTIONARY):
    """ArUco marker centred at (cx, cy), rotated counter-clockwise on the page by 90 deg * quarter_turns."""
    cells = dictionary.markerSize + 2  # data bits + 1-cell border on each side
    bits = cv2.aruco.generateImageMarker(dictionary, marker_id, cells)
    bits = np.rot90(bits, quarter_turns)
    n = bits.shape[0]
    cell = size_mm / n
    black = [(c, n - 1 - r) for r in range(n) for c in range(n) if bits[r, c] == 0]
    fill_cells(ax, black, cx - size_mm / 2, cy - size_mm / 2, cell)


def outline(ax, cx, cy, size_mm):
    ax.add_patch(Rectangle((cx - size_mm / 2, cy - size_mm / 2), size_mm, size_mm, fill=False,
                           edgecolor=GREY, lw=0.4))


def top_label(ax, cx, y_top, text):
    """Label below a cut outline: an arrow showing the marker's top edge (up on the page), then the text."""
    ax.annotate("", xy=(cx - 1, y_top), xytext=(cx - 1, y_top - 7), arrowprops=dict(arrowstyle="-|>", lw=0.8))
    ax.text(cx - 1, y_top - 8.5, text, ha="center", va="top", fontsize=6.5, linespacing=1.3)


def scale_bar(ax, x, y):
    ax.plot([x, x + 100], [y, y], color="black", lw=0.8)
    for xt in (x, x + 100):
        ax.plot([xt, xt], [y - 2, y + 2], color="black", lw=0.8)
    ax.text(x + 50, y + 3, "this bar must measure exactly 100 mm; if not, reprint at 100% / Actual size",
            ha="center", va="bottom", fontsize=7)


def sheet_table_wrist():
    fig, ax = new_page(f"Table marker (ID {C.TABLE_ID}) and hand marker (ID {C.WRIST_ID})")
    t = C.TABLE_MARKER * 1000
    tx, ty = A4[0] / 2, 205
    draw_marker(ax, C.TABLE_ID, tx, ty, t, dictionary=TABLE_DICTIONARY)
    outline(ax, tx, ty, t + 20)
    ax.annotate("", xy=(tx + t / 2 + 22, ty), xytext=(tx + t / 2 + 12, ty),
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.text(tx + t / 2 + 17, ty + 2, "+x", fontsize=8, ha="center")
    top_label(ax, tx, ty - t / 2 - 12, f"ID {C.TABLE_ID} ({t:.0f} mm): arrow = top edge = table +y. Origin at the centre, "
              "z up.\nGlue flat to the table; it must not move during a session.")

    w = C.WRIST_MARKER * 1000
    wx, wy = A4[0] / 2, 95
    draw_marker(ax, C.WRIST_ID, wx, wy, w)
    outline(ax, wx, wy, w + 12)
    top_label(ax, wx, wy - w / 2 - 8, f"ID {C.WRIST_ID} ({w:.0f} mm): arrow = top edge, points toward the FINGERTIPS.\n"
              "Glue onto stiff card, tape to the back of the hand. It must not shift during a clip.")
    scale_bar(ax, 55, 20)
    return fig


CUBE_FACE_NAMES = {1: "top face", 2: "side +x", 3: "side +y", 4: "side -x", 5: "side -y"}


def top_edge_face():
    """The side face that ID 1's top edge points to, from constants.CUBE_MARKER_AXES."""
    y = np.array(C.CUBE_MARKER_AXES[C.CUBE_TOP_ID][1])
    return next(i for i in C.CUBE_SIDE_IDS if np.allclose(C.CUBE_MARKER_AXES[i][2], y))


def cube_diagram(ax, cx, cy, size=34):
    """Top view of the cube with the face IDs and the direction of ID 1's top edge."""
    h = size / 2
    ax.add_patch(Rectangle((cx - h, cy - h), size, size, fill=False, edgecolor="black", lw=0.8))
    for marker_id, (dx, dy) in {2: (1, 0), 3: (0, 1), 4: (-1, 0), 5: (0, -1)}.items():
        ax.text(cx + dx * (h + 5), cy + dy * (h + 5), str(marker_id), ha="center", va="center", fontsize=10,
                weight="bold")
    top_face = top_edge_face()
    sign = 1 if top_face == 2 else -1  # the diagram is drawn with face 2 on the right
    ax.annotate("", xy=(cx + sign * (h - 4), cy), xytext=(cx - sign * (h - 6), cy),
                arrowprops=dict(arrowstyle="-|>", lw=0.8))
    ax.text(cx - 3, cy + 5, f"1 (top):\ntop edge -> {top_face}", ha="center", va="bottom", fontsize=6.5)
    ax.text(cx, cy - h - 12, "cube seen from above", ha="center", va="top", fontsize=7)


def sheet_cube():
    s = C.CUBE_SIDE * 1000
    m = C.CUBE_MARKER * 1000
    fig, ax = new_page(f"Cube markers, IDs 1-5 ({m:.1f} mm markers on {s:.0f} mm faces)")
    xs = (A4[0] / 2 - s - 14, A4[0] / 2, A4[0] / 2 + s + 14)
    layout = {1: (xs[0], 225), 2: (xs[1], 225), 3: (xs[2], 225), 4: (xs[0], 140), 5: (xs[1], 140)}
    for marker_id, (fx, fy) in layout.items():
        draw_marker(ax, marker_id, fx, fy, m)
        outline(ax, fx, fy, s)
        top_label(ax, fx, fy - s / 2 - 2, f"ID {marker_id}, {CUBE_FACE_NAMES[marker_id]}\n(arrow = top edge)")
    cube_diagram(ax, xs[2], 145)
    ax.text(A4[0] / 2, 80,
            "Seen from above, side markers 2 -> 3 -> 4 -> 5 go counter-clockwise. Each side marker is upright: its top\n"
            f"edge touches the top face. ID 1 (top face) has its top edge toward the ID {top_edge_face()} face. "
            "Bottom face: no marker.\n"
            "Cut each square on its outline. If a square overhangs the cube's rounded edges, trim the white margin,\n"
            "never the black. Tape around the middle layer so the Rubik's cube cannot twist.",
            ha="center", va="top", fontsize=7.5, linespacing=1.5)
    scale_bar(ax, 55, 25)
    return fig


def sheet_checkerboard():
    cols, rows = C.CHECKER_INNER[1] + 1, C.CHECKER_INNER[0] + 1  # portrait: 7 x 10 squares
    sq = C.CHECKER_SQUARE * 1000
    fig, ax = new_page(f"Calibration checkerboard: {C.CHECKER_INNER[0]}x{C.CHECKER_INNER[1]} inner corners, "
                       f"{sq:.0f} mm squares")
    x0, y0 = (A4[0] - cols * sq) / 2, 30
    fill_cells(ax, [(c, r) for r in range(rows) for c in range(cols) if (r + c) % 2 == 0], x0, y0, sq)
    ax.text(A4[0] / 2, y0 - 4, "Tape flat to something rigid (a clipboard or a book). Check one square = 25 mm.",
            ha="center", va="top", fontsize=7.5)
    scale_bar(ax, 55, 12)
    return fig


def verify(pdf_paths, dpi=300):
    """Rasterise each PDF and re-detect markers / checkerboard to check the printed geometry."""
    if shutil.which("pdftoppm") is None:
        print("pdftoppm not found; skipping the size check")
        return
    from src.track import detect, make_detector  # the pipeline's own detection, incl. a separate table dictionary

    detector = make_detector()
    with tempfile.TemporaryDirectory() as tmp:
        for pdf in pdf_paths:
            png = Path(tmp) / pdf.stem
            subprocess.run(["pdftoppm", "-r", str(dpi), "-gray", "-png", "-singlefile", str(pdf), str(png)],
                           check=True)
            img = cv2.imread(str(png) + ".png", cv2.IMREAD_GRAYSCALE)
            for marker_id, c in sorted(detect(detector, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)).items()):
                side_mm = np.mean(np.linalg.norm(c - np.roll(c, 1, axis=0), axis=1)) / dpi * 25.4
                expected = C.MARKER_SIZE[int(marker_id)] * 1000
                up = c[0] - c[3]  # marker +y in image pixels (x right, y down)
                print(f"  {pdf.name}: ID {marker_id:2d} side {side_mm:6.2f} mm (expected {expected:6.2f}) "
                      f"{'OK' if abs(side_mm - expected) < 0.3 else 'MISMATCH'}, top edge points "
                      f"{['right', 'down', 'left', 'up'][int(np.round(np.arctan2(up[1], up[0]) / (np.pi / 2))) % 4]}")
            if "checker" in pdf.name:
                found, pts = cv2.findChessboardCorners(img, C.CHECKER_INNER)
                if found:
                    pts = pts.reshape(C.CHECKER_INNER[1], C.CHECKER_INNER[0], 2)
                    step = np.median(np.linalg.norm(np.diff(pts, axis=1), axis=2)) / dpi * 25.4
                    print(f"  {pdf.name}: checkerboard found, square {step:.2f} mm "
                          f"(expected {C.CHECKER_SQUARE * 1000:.2f})")
                else:
                    print(f"  {pdf.name}: checkerboard NOT found")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sheets = {"markers_table_wrist.pdf": sheet_table_wrist, "markers_cube.pdf": sheet_cube,
              "checkerboard.pdf": sheet_checkerboard}
    paths = []
    for name, make in sheets.items():
        fig = make()
        fig.savefig(OUT_DIR / name)  # no bbox_inches="tight": that would crop and change the scale
        plt.close(fig)
        paths.append(OUT_DIR / name)
        print(f"wrote {OUT_DIR / name}")
    verify(paths)


if __name__ == "__main__":
    main()
