# Sprint 3 — porównanie modeli

## Wyniki

Modele oceniono na tym samym zbiorze testowym Iranian Churn: 570 rekordów,
w tym 89 przypadków `Churn = 1`. Główna metryka to recall dla klasy pozytywnej.

| Model | Recall | Mediana predykcji 1 rekordu (ms) | Rozmiar artefaktu (B) | Przyrost RAM po wczytaniu (MiB) |
| --- | ---: | ---: | ---: | ---: |
| Regresja logistyczna (baseline) | 47,19% | 1,816 | 3 919 | 0,066 |
| AutoGluon LightGBM | 75,28% | 3,359 | 305 140 | 17,895 |
| AutoGluon CatBoost | 86,52% | 3,956 | 808 840 | 25,250 |

CatBoost wykrył 77 z 89 odchodzących klientów, a LightGBM wykrył 67.
LightGBM wykonywał predykcję szybciej, zajmował mniej miejsca na dysku
i wymagał mniejszego przyrostu RAM-u po wczytaniu. Regresja logistyczna
miała najmniejszy artefakt i najkrótszy czas predykcji.

## Metoda pomiaru

- **Recall** pochodzi z lokalnego raportu `data/08_reporting/sprint3_results.json`.
  Wyniki po zaokrągleniu zgadzają się z runami W&B opisanymi w
  [PR #13](https://github.com/madziasroka-git/asi-projekt/pull/13).
- **Czas predykcji** to mediana 50 wywołań `predict` na tym samym pierwszym
  rekordzie testowym, po jednej predykcji na rozgrzewkę. Kolumny kategoryczne
  dla AutoGluona przekonwertowano przed pomiarem.
- **Rozmiar artefaktu** to rozmiar pliku `model.pkl` dla regresji logistycznej
  oraz suma rozmiarów wszystkich plików w katalogu danego predyktora AutoGluona.
- **Przyrost RAM-u** to różnica pamięci rezydentnej procesu (RSS) przed
  wczytaniem modelu i bezpośrednio po nim. Każdy model mierzono w osobnym
  procesie. Warianty AutoGluona wczytano do pamięci przez `persist()`.

Pomiary wykonano na jednym komputerze z Windows 11, w środowisku `asi-ml`
z Pythonem 3.11.18. Czasy i przyrost RSS mogą się różnić między komputerami
i powtórzeniami.

- Git SHA treningu: `91188ea1495980aea99472dd43f1b4a3bba0ccd1`
- SHA-256 danych: `90d5fb6bd1630cd4de4b4d28fcf8b4cb92a8f6ab7484605b0799d47386f7dbe1`

## Odtworzenie pomiarów

Uruchom z głównego katalogu projektu, w aktywnym `asi-ml` i z pobranym pełnym
zbiorem danych.

```bat
set "WANDB_MODE=disabled"
python -m kedro run --pipelines experiments
python scripts\benchmark_models.py logistic_regression
python scripts\benchmark_models.py autogluon_lightgbm
python scripts\benchmark_models.py autogluon_catboost
```

Ścieżki do artefaktów są w polach `model_path` raportu. Na Windowsie rozmiar
pliku można odczytać przez `dir /-c`, a sumę rozmiarów plików katalogu przez
`dir /s /a:-d /-c`.
