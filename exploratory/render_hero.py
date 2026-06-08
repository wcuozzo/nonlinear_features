"""Regen hero with cleaner title (no 'seed 2' prefix)."""
import colorsys, numpy as np, torch, matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import core
from core import Autoencoder

device = torch.device('cpu'); core.device = device

n, m, l, S = 64, 2, 4, 0.95
EXT = 1.0
GRID = 800

def sci_palette(n, sat=0.7, val=0.88, scramble=37):
    arr = np.zeros((n, 3))
    for i in range(n):
        hue = ((i * scramble) % n) / n
        v = val + 0.04 * ((i // 8) % 3 - 1)
        s = sat + 0.05 * ((i // 5) % 3 - 1)
        arr[i] = colorsys.hsv_to_rgb(hue, s, v)
    return arr

palette = sci_palette(n)
cmap = mcolors.ListedColormap(palette)

model = Autoencoder(n, m, l, tied_weights=False)
model.load_state_dict(torch.load('exploratory/seed_models/seed2.pt'))
model.eval()

xs_g = np.linspace(-EXT, EXT, GRID)
XX, YY = np.meshgrid(xs_g, xs_g, indexing='xy')
Z = np.stack([XX, YY], axis=-1).reshape(-1, 2)
with torch.no_grad():
    x_hat = model.decode(torch.tensor(Z, dtype=torch.float32)).numpy()
a = np.maximum(x_hat, 0)
argmax = a.argmax(-1).reshape(GRID, GRID)
dec_mag = np.log1p(np.linalg.norm(x_hat, axis=1)).reshape(GRID, GRID)

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(argmax, origin='lower', extent=[-EXT, EXT, -EXT, EXT],
          cmap=cmap, vmin=0, vmax=n - 1, aspect='equal',
          interpolation='nearest')
levels = np.linspace(dec_mag.min(), dec_mag.max(), 18)[2:-1]
ax.contour(XX, YY, dec_mag, levels=levels, colors='#222',
           linewidths=0.6, alpha=0.75)
ax.set_xlim(-EXT, EXT); ax.set_ylim(-EXT, EXT)
ax.set_xlabel(r'$z_1$', fontsize=10)
ax.set_ylabel(r'$z_2$', fontsize=10)
ax.set_title(f'n={n}, m={m}, l={l}, S={S}', fontsize=10)
ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig('fig_hero_argmax_regions.png', dpi=130, bbox_inches='tight')
print('Wrote fig_hero_argmax_regions.png')
