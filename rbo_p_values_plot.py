# Code to generate a Plot showing the impact of different p values at certain depths
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt


def rbo_contribution(p, depth):
    return p ** np.arange(depth)

depth = 50
p_values = [0.99, 0.98, 0.95, 0.9]

plt.figure(figsize=(10, 6))

for p in p_values:
    contributions = rbo_contribution(p, depth)
    plt.plot(range(1, depth + 1), contributions, label=f'p = {p}', lw=2)

plt.title('RBO Contribution per Rank for Different p Values', fontsize=14)
plt.xlabel('Rank Position', fontsize=12)
plt.ylabel('Unscaled Contribution to RBO Score', fontsize=12)
plt.legend(title='p Value', fontsize=12)
plt.grid(True)
plt.tight_layout()

plt.savefig('./Plots/rbo_contribution_plot.png')