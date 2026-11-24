# Model Card

## Model

- Typ modelu i jego wersja (alias `production` w W&B): AutoGluon Tabular z modelem CatBoost
  (`learning_rate: 0.05`, `depth: 6`, limit czasu 120 s). Artefakt W&B `iranian-churn-model`
  z aliasem `production`.
- Data utworzenia i autor runu: 17.11.2026, Aleksandra Sutyniec (pipeline `experiments`).
  Pakiet modelu: Leonard Hołubowicz.

- Jak użyć: `load_model_package()` i `predict()` z `src/asi_project/model_package.py`.
  Wynik ma kolumny `prediction` (0 albo 1) i `probability` (prawdopodobieństwo klasy `1`).
  Sprawdzenie: `python scripts/smoke_model.py`.

## Przeznaczenie

- Problem (binary classification) i klasa pozytywna: przewidujemy, czy klient operatora
  telekomunikacyjnego odejdzie (`Churn`). Klasa pozytywna: `1` (klient odszedł).
- Użytkownik systemu i decyzja, którą wspiera model: dział obsługi klienta. Model wskazuje
  klientów zagrożonych odejściem na koniec 9. miesiąca, żeby można było zaproponować im
  działania retencyjne przed 12. miesiącem.
- Zastosowania, do których model nie jest przeznaczony: automatyczne decyzje o kliencie
  bez udziału człowieka (np. zmiana ceny, blokada usługi), dane innych operatorów lub krajów,
  predykcja w innym momencie niż koniec 9. miesiąca.

## Dane

- Dataset (id z `docs/datasets.md`) i fingerprint danych: `iranian_churn`, SHA-256
  `90d5fb6bd1630cd4de4b4d28fcf8b4cb92a8f6ab7484605b0799d47386f7dbe1`.
- Strategia splitu z kontraktu i liczebności zbiorów: `stratified_random`, `test_size: 0.2`,
  `random_state: 42`. Po usunięciu duplikatów 2850 rekordów: 2280 treningowych i 570 testowych
  (w tym 89 z `Churn = 1`).
- Cechy (11): `Call Failure`, `Subscription Length`, `Seconds of Use`, `Frequency of use`,
  `Frequency of SMS`, `Distinct Called Numbers`, `Complains`, `Tariff Plan`, `Status`,
  `Age Group`, `Charge Amount`.

## Metryki

- Metryka główna na zbiorze testowym, w porównaniu z DummyClassifier: recall dla klasy `1`
  wynosi 86,52%. DummyClassifier zawsze przewiduje `0`, więc ma recall 0%.
- Pozostałe metryki i macierz pomyłek: precision 90,59%, F1 88,51%, accuracy 96,49%.
  Macierz pomyłek: TN 473, FP 8, FN 12, TP 77. Baseline (regresja logistyczna) miał
  recall 47,19%.

## Ograniczenia i ryzyka

- Ograniczenia danych (z `docs/datasets.md`): mały zbiór (3150 rekordów) z jednego operatora
  w Iranie. Tylko 15,71% klientów odchodzi. Część kategorii jest zakodowana liczbami
  (`Charge Amount`, `Age Group`, `Tariff Plan`, `Status`). Kolumna `Customer Value` jest
  wykluczona przez kontrakt.
- Znane błędy i sytuacje, w których model zawodzi: model nie wykrywa 12 z 89 odchodzących
  klientów w zbiorze testowym i fałszywie wskazuje 8 klientów. Model jest większy i wolniejszy
  niż baseline (ok. 0,8 MB, ok. 4 ms na rekord, ok. 25 MiB RAM). Pomiary wykonano na jednym
  komputerze.

## Wersje

- Wersje bibliotek modelu i git SHA treningu: Python 3.11, autogluon.tabular 1.6.3,
  scikit-learn 1.7.2, pandas 2.3.3, numpy 2.2.6. Git SHA treningu:
  `91188ea1495980aea99472dd43f1b4a3bba0ccd1`. Pełna lista wersji jest w `metadata.json`
  pakietu modelu.
