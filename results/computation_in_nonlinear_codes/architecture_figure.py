"""Render the README diagram of the trained XOR network; run from any directory.

Layer sizes match results/joint_decoder_refinement/selected-example-exact.json:
encoder 4 -> 4 -> 4 -> 2, four residual blocks with two ReLU units each,
a 2 -> 2 XOR head, and decoders 2 -> 4 -> 4 -> 4.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, GREY, DARK = "#245589", "#91979d", "#30363b"
C_IN, C_HID, C_Z, C_R, C_OUT = "#a15cb2", "#5095d1", "#d86d49", "#e3a33b", "#5cb461"


def col(n, center, gap=0.62):
    return center + gap * (np.arange(n) - (n - 1) / 2)


def main():
    root = Path(__file__).resolve().parents[2]
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "cm"})
    # A compact canvas keeps text legible when GitHub scales the image to
    # the README column width.
    fig, ax = plt.subplots(figsize=(11, 5.6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(-0.6, 14.2), ylim=(-0.6, 8.3))
    ax.axis("off")

    def dense(xa, ya, xb, yb, color=GREY, alpha=0.32, lw=0.8):
        for y1 in ya:
            for y2 in yb:
                ax.plot([xa, xb], [y1, y2], color=color, alpha=alpha, lw=lw,
                        zorder=1)

    def nodes(x, yy, color):
        ax.scatter(np.full(len(yy), x), yy, s=300, facecolor=color,
                   edgecolor=DARK, linewidth=1.1, zorder=3)

    main_y, relu_y, low_y = 5.0, 6.55, 1.6  # main path, ReLU branch, decoders

    # Encoder.
    x_in, x_e1, x_e2, x_z0 = 0.3, 1.6, 2.9, 4.2
    ys4m, ys2m = col(4, main_y), col(2, main_y, 0.7)
    dense(x_in, ys4m, x_e1, ys4m)
    dense(x_e1, ys4m, x_e2, ys4m)
    dense(x_e2, ys4m, x_z0, ys2m)

    # Residual blocks: ReLU branch above the main row, identity skip along it.
    zx = [x_z0]
    step = 1.55
    r_ys = col(2, relu_y, 0.6)
    for _ in range(4):
        xr, xz = zx[-1] + step / 2, zx[-1] + step
        dense(zx[-1], ys2m, xr, r_ys, color=C_R, alpha=0.55, lw=1.0)
        dense(xr, r_ys, xz, ys2m, color=C_R, alpha=0.55, lw=1.0)
        for y in ys2m:
            ax.plot([zx[-1], xz], [y, y], color=DARK, alpha=0.55, lw=1.2,
                    zorder=1)
        nodes(xr, r_ys, C_R)
        zx.append(xz)
    x_z4 = zx[-1]

    # XOR head.
    x_head = x_z4 + 1.25
    dense(x_z4, ys2m, x_head, ys2m)

    # Decoders.
    ys4l = col(4, low_y)

    def decoder(x0, xz):
        xs = [x0, x0 + 1.3, x0 + 2.6]
        dense(xz, ys2m, xs[0], ys4l, alpha=0.25)
        dense(xs[0], ys4l, xs[1], ys4l)
        dense(xs[1], ys4l, xs[2], ys4l)
        nodes(xs[0], ys4l, C_HID)
        nodes(xs[1], ys4l, C_HID)
        nodes(xs[2], ys4l, C_OUT)
        return xs

    dec0 = decoder(x_z0 + 0.9, x_z0)
    dec4 = decoder(x_z4 + 0.55, x_z4)

    for x, yy, c in [(x_in, ys4m, C_IN), (x_e1, ys4m, C_HID),
                     (x_e2, ys4m, C_HID), (x_head, ys2m, C_OUT)]:
        nodes(x, yy, c)
    for x in zx:
        nodes(x, ys2m, C_Z)

    # Labels.
    def bracket(x1, x2, y, label, color=BLUE, up=True):
        d = -0.14 if up else 0.14
        ax.plot([x1, x1, x2, x2], [y + d, y, y, y + d], color=color, lw=1.4)
        ax.text((x1 + x2) / 2, y + (0.18 if up else -0.2), label,
                ha="center", va="bottom" if up else "top", fontsize=15,
                weight="bold", color=color)

    bracket(x_in, x_z0, 7.6, "Encoder (frozen)")
    bracket(x_z0 + 0.15, x_z4, 7.6, "Four residual ReLU blocks", color=C_R)
    ax.text(x_head, 7.75, "XOR head", ha="center", fontsize=15,
            weight="bold", color="#3b8a3f")
    bracket(dec0[0], dec0[2], 0.2, "Original decoder", up=False)
    bracket(dec4[0], dec4[2], 0.2, "New decoder", up=False)

    def under(x, y, text, size=12.5):
        ax.text(x, y, text, ha="center", va="top", fontsize=size,
                color="#22272c", linespacing=1.3)

    yl = main_y - 1.55
    under(x_in, yl, "Input\n" + r"$x \in \{0,1\}^4$")
    under(x_e1, yl, "Hidden")
    under(x_e2, yl, "Hidden")
    for i, x in enumerate(zx):
        under(x, yl, f"$z_{i}$")
    ax.text(x_head + 0.3, main_y, "XOR\noutputs", ha="left", va="center",
            fontsize=12.5, color="#22272c")
    for xs in (dec0, dec4):
        under(xs[0], low_y + 1.52, "Hidden", 11)
        under(xs[1], low_y + 1.52, "Hidden", 11)
        under(xs[2], low_y + 1.52, "Bits", 11)
    ax.text(zx[0] + 0.1, relu_y + 0.7, r"ReLU branch $r_{i+1}$", ha="left",
            va="center", fontsize=11, color="#9a6a17")
    ax.plot([zx[0] + 2.55, zx[0] + 3.05], [relu_y + 0.7] * 2, color=DARK,
            lw=1.2, alpha=0.7)
    ax.text(zx[0] + 3.15, relu_y + 0.7,
            r"skip: $z_{i+1} = z_i + B_i r_{i+1} + b_i$", va="center",
            fontsize=11, color="#454e57")

    fig.savefig(root / "figures/fig_trained_xor_architecture.png", dpi=240,
                facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


if __name__ == "__main__":
    main()
