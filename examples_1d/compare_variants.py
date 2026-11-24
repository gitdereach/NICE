"""Compare identification of E across dissipation-penalty and loop variants (all with 2 % noise)."""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
E_REF = 2.0e5
V = [('noise2', 'bazowy\n(kara sq, w=1)'), ('pen_l1_w1', 'kara L1\nw=1'), ('pen_l1_w10', 'kara L1\nw=10'),
     ('loop_sq_w1', 'pętla\n+ kara sq'), ('loop_l1_w1', 'pętla\n+ L1 w=1'), ('loop_l1_w10', 'pętla\n+ L1 w=10')]
R = {t: json.load(open(os.path.join(OUT, f'results_{t}.json'))) for t, _ in V}

fig, ax = plt.subplots(1, 3, figsize=(18, 4.5), dpi=150)
x = np.arange(len(V))
med = np.array([R[t]['E_median'] for t, _ in V]) / 1e3
lo = np.array([R[t]['E_p5_p95'][0] for t, _ in V]) / 1e3
hi = np.array([R[t]['E_p5_p95'][1] for t, _ in V]) / 1e3
c = ['gray', 'tab:blue', 'tab:blue', 'tab:green', 'tab:purple', 'tab:purple']
ax[0].bar(x, med, color=c, alpha=.8)
ax[0].errorbar(x, med, yerr=[med - lo, hi - med], fmt='none', ecolor='k', capsize=4)
ax[0].axhline(E_REF / 1e3, color='r', lw=2, label='prawdziwe E = 200 000')
ax[0].set_xticks(x); ax[0].set_xticklabels([l for _, l in V], fontsize=8)
ax[0].set_ylabel('E [×10³] (mediana, zakres 5–95%)'); ax[0].set_title('Wyznaczona sztywność'); ax[0].legend(fontsize=8)

err = [R[t]['test_cyclic_rel_rmse_sigma'] * 100 for t, _ in V]
neg = [R[t]['negative_dissipation_fraction_cyclic'] * 100 for t, _ in V]
ax[1].bar(x - 0.2, err, 0.4, label='błąd σ, ścieżki cykliczne [%]', color='tab:blue')
ax[1].bar(x + 0.2, neg, 0.4, label='punkty z ujemną dyssypacją [%]', color='tab:orange')
ax[1].set_xticks(x); ax[1].set_xticklabels([l for _, l in V], fontsize=8); ax[1].legend(fontsize=8)
ax[1].set_title('Dokładność i II zasada termodynamiki')

for t, lab, col in [('noise2', 'bazowy', 'gray'), ('loop_l1_w10', 'pętla + L1 w=10', 'tab:purple')]:
    cc = R[t]['curves']['cyclic'][0]
    ax[2].plot(np.array(cc['eps']) * 100, np.array(cc['eps_e_pred']) * 1e3, color=col, lw=1.5, label=f'odkryte εᵉ: {lab}')
cc = R['noise2']['curves']['cyclic'][0]
ax[2].plot(np.array(cc['eps']) * 100, np.array(cc['sigma_ref']) / E_REF * 1e3, 'r--', lw=2, label='prawdziwe εᵉ = σ/E')
ax[2].set_xlabel('ε [%]'); ax[2].set_ylabel('εᵉ [‰]'); ax[2].set_title('Odkształcenie sprężyste (ścieżka cykliczna)')
ax[2].legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'compare_variants.png'))

rows = ['| wariant (szum 2 %) | E (mediana) | εᵉ: skala vs prawda | błąd σ cykle | błąd σ obc.–odc. | Rp0.2 | ujemna d |',
        '|---|---|---|---|---|---|---|']
for t, l in V:
    r = R[t]
    rows.append(f"| {l.replace(chr(10), ' ')} | {r['E_median']:,.0f} ({r['E_median'] / E_REF - 1:+.0%}) | "
                f"{1 / r['eps_e_vs_true']['slope']:.2f}× | {r['test_cyclic_rel_rmse_sigma']:.2%} | "
                f"{r['test_loading_unloading_rel_rmse_sigma']:.2%} | {r['Rp0.2']:.1f} | {r['negative_dissipation_fraction_cyclic']:.1%} |")
rows.append(f"| referencja | 200 000 | 1.00× | – | – | {R['noise2']['Rp0.2_ref']:.1f} | 0 % |")
table = '\n'.join(rows)
open(os.path.join(OUT, 'compare_variants.md'), 'w').write(table + '\n')
print(table)
