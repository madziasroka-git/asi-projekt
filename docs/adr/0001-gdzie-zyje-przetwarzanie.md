# ADR-1: Gdzie żyje przetwarzanie danych

- **Status:** proponowany
- **Sprint:** S2
- **Autor:** Jan Bagiński
- **Recenzent:** Aleksandra Sutyniec

## Kontekst

W Sprincie 2 przygotowujemy dane Iranian Churn i trenujemy model przez Kedro.
Domyślny przebieg łączy `data_processing` i `data_science`
([rejestr pipeline’ów](../../src/asi_project/pipeline_registry.py)).
W przyszłym API model będzie otrzymywał pojedynczy rekord klienta, dla którego
nie znamy jeszcze etykiety `Churn`. Trzeba ustalić, które kroki przygotowania
danych mogą być wspólne dla treningu i predykcji.

## Rozważone opcje

1. Wykonać wszystkie operacje w `data_processing` przed podziałem train/test,
   w tym dopasowanie imputacji, skalowania i kodowania kategorii.
2. Oddzielić normalizację pojedynczego rekordu i filtrowanie danych od
   transformacji dopasowywanych podczas treningu modelu.

## Decyzja

Wybieramy opcję 2:

- **Normalizacja wejścia** pozostaje w `normalize_input()` w
  [`data.py`](../../src/asi_project/data.py). Funkcja przycina spacje w tekście
  i zamienia puste napisy na braki. Nie potrzebuje etykiety ani statystyk
  zbioru, więc ta sama funkcja może być używana podczas treningu i w
  [przyszłym API](../sprints/SPRINT_4.md).
- **Filtrowanie rekordów** wykonuje `filter_training_rows()` w tym samym pliku.
  Usuwa rekordy bez etykiety i dokładne duplikaty całych wierszy. W Kedro
  [`data_processing`](../../src/asi_project/pipelines/data_processing/pipeline.py)
  wykonuje filtrowanie po normalizacji, a
  [`data_science`](../../src/asi_project/pipelines/data_science/pipeline.py)
  dzieli dopiero otrzymane `processed_data` na train i test. Tego filtrowania
  nie wykonujemy przy predykcji dla klienta.
- **Przetwarzanie cech modelu** pozostaje w sklearn `Pipeline` z
  [`model.py`](../../src/asi_project/model.py). Cechy liczbowe mają imputację
  medianą i skalowanie, a kategoryczne — imputację najczęstszą wartością
  i kodowanie one-hot. Node treningowy wywołuje `model.fit(X_train, y_train)`
  ([kod noda](../../src/asi_project/pipelines/data_science/nodes.py)), więc
  transformacje są dopasowane na danych treningowych. Przy ocenie model
  stosuje je do `X_test` bez ponownego dopasowania.

| Wielkość | Wartość | Skąd pochodzi |
|---|---:|---|
| Liczba rekordów przed filtrowaniem i po nim | 3150 → 2850 | Sprawdzenie `data_processing` na pełnym zbiorze Iranian Churn |
| Udział zbioru testowego | 20% | `split.test_size: 0.2` w [`conf/base/parameters.yml`](../../conf/base/parameters.yml) |

## Odrzucone alternatywy

Odrzucamy dopasowanie transformacji przed podziałem train/test. Przykładowo
mediana do imputacji wyliczona z całego zbioru uwzględniałaby też dane testowe.
Model korzystałby więc podczas treningu z informacji o zbiorze, na którym
ma być później oceniany.

## Konsekwencje

[Data Catalog](../../conf/base/catalog.yml) zapisuje cały wytrenowany
sklearn `Pipeline` w `data/06_models/model.pkl`. Dzięki temu klasyfikator
i dopasowane transformacje są przechowywane razem. W Sprincie 4 API wykona
normalizację wejścia i przekaże rekord do wczytanego modelu, bez ponownej
implementacji imputacji, skalowania i kodowania kategorii.

## Historia

| Data | Zmiana |
|---|---|
| 2026-11-11 | Utworzenie ADR-1 |
