"""Summary figure and table for examples_1d/nice_1d_hypoplastic.py runs (noise 0, 2, 5 %)."""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
runs = {n: json.load(open(os.path.join(OUT, f'results_noise{n}.json'))) for n in (0, 2, 5)}
col = {0: 'darkblue', 2: 'tab:green', 5: 'tab:orange'}

fig, ax = plt.subplots(1, 4, figsize=(20, 4.3), dpi=150)
tr = runs[5]['curves']['train_example']
e = np.array(tr['eps']) * 100
ax[0].plot(e, tr['sigma_ref'], 'r-', lw=3, alpha=.5, label='prawdziwa odpowiedź')
ax[0].plot(e[tr['samp_idx']], tr['sigma_meas'], 'ko', ms=3, label='40 pomiarów (szum 5%)')
ax[0].set_title('Dane treningowe: obciążenie–odciążenie'); ax[0].set_xlabel('ε [%]'); ax[0].set_ylabel('σ')

for k, i in enumerate((0, 1)):
    c0 = runs[0]['curves']['cyclic'][i]
    a = ax[1 + k]
    a.plot(np.array(c0['eps']) * 100, c0['sigma_ref'], 'r-', lw=3, alpha=.5, label='referencja')
    for n, r in runs.items():
        c = r['curves']['cyclic'][i]
        a.plot(np.array(c['eps']) * 100, c['sigma_pred'], '-', lw=1.2, color=col[n], label=f'NICE, szum {n}%')
    a.set_title(f'Test: ścieżka cykliczna {i + 1} (niewidziana)'); a.set_xlabel('ε [%]'); a.set_ylabel('σ')

m = runs[0]['curves']['monotonic']
ax[3].plot(np.array(m['eps']) * 100, m['sigma_ref'], 'r-', lw=3, alpha=.5, label='referencja')
for n, r in runs.items():
    mm = r['curves']['monotonic']
    ax[3].plot(np.array(mm['eps']) * 100, mm['sigma_pred'], '-', color=col[n], label=f'NICE, szum {n}%')
ax[3].set_title('Rozciąganie monotoniczne'); ax[3].set_xlabel('ε [%]'); ax[3].set_ylabel('σ')
for a in ax:
    a.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'summary.png'))

rows = ['| | szum 0% | szum 2% | szum 5% | referencja |', '|---|---|---|---|---|']
f = lambda key, fmt='{:.2%}': ' | '.join(fmt.format(runs[n][key]) for n in (0, 2, 5))
rows.append(f"| błąd σ, testy obciążenie–odciążenie | {f('test_loading_unloading_rel_rmse_sigma')} | – |")
rows.append(f"| błąd σ, ścieżki cykliczne (niewidziane) | {f('test_cyclic_rel_rmse_sigma')} | – |")
rows.append('| εᵉ odkryte vs σ/E (R²) | ' + ' | '.join(f"{runs[n]['eps_e_vs_true']['R2_linear']:.4f}" for n in (0, 2, 5)) + ' | 1 |')
rows.append(f"| E (mediana d²u/dεᵉ²) | {f('E_median', '{:,.0f}')} | 200 000 |")
rows.append(f"| Rp0.2 | {f('Rp0.2', '{:.1f}')} | {runs[0]['Rp0.2_ref']:.1f} |")
rows.append(f"| σ przy ε = 0.6% | {f('sigma_at_0.6pct', '{:.1f}')} | {runs[0]['sigma_at_0.6pct_ref']:.1f} |")
rows.append(f"| ujemna dyssypacja (ułamek punktów) | {f('negative_dissipation_fraction_cyclic', '{:.1%}')} | 0 |")
table = '\n'.join(rows)
open(os.path.join(OUT, 'summary_table.md'), 'w').write(table + '\n')
print(table)
