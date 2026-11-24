# Odtworzenie przykładu 1 (Benchmark 1) z artykułu NICE

Masi & Einav (2024), *Neural integration for constitutive equations using small data*, CMAME 420, 116698.
Materiał: sprężysto-plastyczny Drucker–Prager (K = 70 GPa, G = 60 GPa); dane: pojedyncze ścieżki
obciążenie–odciążenie, 40 punktów naprężenia na ścieżkę, 24 ścieżki trening+walidacja.

```bash
pip install torch torchdiffeq scipy matplotlib nbformat nbclient ipykernel
python reproduction/run_benchmark1.py --mode pretrained              # wagi autorów (~15 s)
python reproduction/run_benchmark1.py --mode train --max-epochs 5000 # trening od zera
```

Skrypt uruchamia oryginalny notebook `benchmarks/[NICE] Benchmark-1.ipynb` bez jego zmieniania
(podmienia w locie tylko komórkę treningu / limit epok / ścieżki zapisu wag).

## Wyniki (wagi autorów)

| MAE naprężenia (znormalizowane) | Notebook autorów | Odtworzenie |
|---|---|---|
| trening + walidacja | 0.0058 | 0.0058 |
| test | 0.0048 | 0.0048 |

Wykresy: `output/pretrained/summary.png` (drenowany test trójosiowy z cyklami – protokół niewidziany w treningu).

## Trening od zera

Autorzy trenowali 67 140 epok (early stopping). Na CPU (4 rdzenie) jedna epoka trwa ~0.7 s, więc pełny trening
to ~13 h. Kontrolnie: po 200 epokach MAE test = 0.096.

## Zmiany w kodzie autorów

- `nice/nice_module.py`, `benchmarks/nice_module.py`: `np.Inf` → `np.inf` (zgodność z NumPy ≥ 2.0).
