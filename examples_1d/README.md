# NICE dla materiału sprężysto-hipoplastycznego 1D w warunkach „laboratoryjnych”

Adaptacja 1D metody NICE (Masi & Einav 2024; `NICE_reduced` z `nice/nice_module.py` jest napisane na sztywno dla 2
składowych p, q). Materiał referencyjny – ten sam co w repozytorium TANN (E = 200 000, S = 141.42):
σ = E·εᵉ, ε̇ᵉ = ε̇ − |ε̇|·σ/S.

**Dane treningowe jak z maszyny wytrzymałościowej**: 24 testy obciążenie → odciążenie do σ = 0 (12 amplitud
0.05–0.6 %, rozciąganie i ściskanie; 14 trening / 5 walidacja / 5 test), stała prędkość odkształcenia,
**40 pomiarów naprężenia na test**, szum gaussowski 0 / 2 / 5 %. Bez εᵉ, ε_pl, energii i dyssypacji. **Bez cykli.**
Test: niewidziane testy obciążenie–odciążenie oraz 5 losowych ścieżek z wieloma cyklami.

```bash
python examples_1d/nice_1d_hypoplastic.py --noise 0 --epochs 5000   # ~20 min CPU (1 wątek)
python examples_1d/plot_results.py
```

## Wyniki (5000 epok)

| | szum 0% | szum 2% | szum 5% | referencja |
|---|---|---|---|---|
| błąd σ, testy obciążenie–odciążenie | 0.89% | 0.86% | 0.98% | – |
| błąd σ, ścieżki cykliczne (niewidziane) | 0.73% | 0.68% | 0.75% | – |
| εᵉ odkryte vs σ/E (R²) | 0.9994 | 0.9995 | 0.9995 | 1 |
| E (mediana d²u/dεᵉ²) | 92,566 | 95,048 | 93,361 | 200 000 |
| Rp0.2 | 140.8 | 140.5 | 140.4 | 138.3 |
| σ przy ε = 0.6% | 142.3 | 142.1 | 141.9 | 141.4 |
| ujemna dyssypacja (ułamek punktów) | 6.6% | 6.4% | 6.7% | 0 |

Błąd σ = RMSE / S. Wykres: `output/summary.png`.

## Wnioski

- **Naprężenia: bardzo dobrze**, także dla ścieżek z wieloma cyklami, których nie było w treningu (błąd < 1 %),
  i praktycznie niezależnie od szumu do 5 % (zgodnie z obserwacją autorów: błąd < poziomu szumu).
- **Zmienna stanu odkryta poprawnie co do kształtu** (R² = 0.9995 względem prawdziwego εᵉ), ale **w złej skali**:
  odkryte εᵉ ≈ 1.9 × prawdziwe, a więc **E wychodzi ≈ 2 × za małe** (~93 000 zamiast 200 000).
  Przyczyna: przekształcenie εᵉ → c·εᵉ, u → c·u daje identyczne naprężenia; jedynie II zasada (d ≥ 0) wyklucza c ≠ 1,
  ale naruszenie jest ∝ σ i bardzo małe przy σ → 0 (stąd ~6.5 % punktów z lekko ujemną dyssypacją).
  W tym materiale nie ma zakresu czysto sprężystego (płynięcie od początku), więc z samych krzywych σ–ε E jest słabo
  identyfikowalne. W benchmarku autorów (Drucker–Prager) odciążenie jest czysto sprężyste i E jest określone.
- **Rp0.2 i σ graniczne** – dobrze (różnica ~2 % i < 1 %).
- Praktycznie: E warto wyznaczyć niezależnie (np. pomiar ultradźwiękowy, mała pętla odciążenia w zakresie sprężystym)
  albo wzmocnić karę za ujemną dyssypację.
