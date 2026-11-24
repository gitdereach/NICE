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

## Identyfikacja E: silniejsza II zasada (krok 1) i mała pętla odciążenia (krok 2)

Dlaczego E jest słabo określone: dla tego materiału przeskalowanie εᵉ → c·εᵉ, u → c·u daje **dokładnie te same
naprężenia** dla dowolnej historii obciążenia. Jedyną informacją, która wymusza c = 1, jest II zasada
(d = σ(ε̇ − ε̇ᵉ) ≥ 0): przy σ → 0 obciążanie wymaga c ≤ 1, a odciążanie c ≥ 1. Naruszenie jest ∝ σ, więc kara
w postaci z repozytorium NICE, mean((relu(−d)/max|d|)²), jest zbyt słaba.

```bash
# krok 1: kara L1 za ujemną dyssypację
python examples_1d/nice_1d_hypoplastic.py --noise 2 --diss-form l1 --w-diss 10 --tag pen_l1_w10
# krok 2: + mała pętla odciążenie–obciążenie (0.05 %) w 40 % obciążenia, 10 dodatkowych pomiarów w pętli
python examples_1d/nice_1d_hypoplastic.py --noise 2 --loop 0.0005 --diss-form l1 --w-diss 10 --tag loop_l1_w10
python examples_1d/compare_variants.py
```

Wszystkie warianty: szum 2 %, 5000 epok.

| wariant (szum 2 %) | E (mediana) | εᵉ: skala vs prawda | błąd σ cykle | błąd σ obc.–odc. | Rp0.2 | ujemna d |
|---|---|---|---|---|---|---|
| bazowy (kara sq, w=1) | 95,048 (-52%) | 1.83× | 0.68% | 0.86% | 140.5 | 6.4% |
| kara L1 w=1 | 170,052 (-15%) | 1.24× | 0.79% | 0.83% | 138.4 | 2.0% |
| kara L1 w=10 | 184,420 (-8%) | 1.13× | 1.44% | 1.46% | 138.3 | 0.7% |
| pętla + kara sq | 108,043 (-46%) | 1.80× | 0.94% | 0.65% | 141.2 | 6.4% |
| pętla + L1 w=1 | 191,993 (-4%) | 1.14× | 0.26% | 0.41% | 138.5 | 1.2% |
| pętla + L1 w=10 | 197,331 (-1%) | 1.04× | 0.44% | 0.38% | 138.5 | 0.1% |
| referencja | 200 000 | 1.00× | – | – | 138.3 | 0 % |

Za silne kary (sq w = 1000, L1 w = 100) psują trening (błąd σ na cyklach 64 % i 43 %) – wyniki w `output/`.

Wnioski:
- **Sama pętla nie pomaga** (E 95 000 → 108 000) – zgodnie z teorią, bo symetria skali jest dokładna dla każdej
  ścieżki obciążenia.
- **Sama kara L1** poprawia E do −8 % (w = 10), kosztem nieco większego błędu σ.
- **Pętla + kara L1 działają razem najlepiej**: E = 197 300 (−1 %), skala εᵉ 1.04×, błąd σ na niewidzianych
  cyklach 0.44 %, ujemna dyssypacja w 0.1 % punktów. Pętla daje dodatkowe stany z odciążaniem przy niezerowym σ,
  w których II zasada jest „aktywna”, a kara L1 sprawia, że te stany realnie wpływają na trening.

Wykres: `output/compare_variants.png`.
