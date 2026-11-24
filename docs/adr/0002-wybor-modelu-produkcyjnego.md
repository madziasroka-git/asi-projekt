# ADR-2: Wybór modelu produkcyjnego i jego postać

- **Status:** proponowany
- **Sprint:** S3
- **Autor:** Magdalena Sroka
- **Recenzent:** imię i nazwisko

## Kontekst

W Sprincie 3 porównaliśmy baseline ze Sprintu 2 i dwie konfiguracje AutoGluona na zbiorze
Iranian Churn. Trzeba wybrać jeden model, który trafi do API w Sprincie 4, i ustalić,
w jakiej postaci go przekazujemy. Główną metryką projektu jest recall dla klasy `1`,
bo chcemy wykryć jak najwięcej klientów, którzy odejdą.

## Rozważone opcje

1. Regresja logistyczna (baseline).
2. AutoGluon LightGBM (`learning_rate: 0.05`, `num_leaves: 31`, limit 120 s).
3. AutoGluon CatBoost (`learning_rate: 0.05`, `depth: 6`, limit 120 s).

Kryteria wyboru:

1. Recall dla klasy `1` na zbiorze testowym (najważniejsze).
2. Czas predykcji jednego rekordu.
3. Rozmiar artefaktu.
4. Pamięć potrzebna po wczytaniu modelu.

## Decyzja

Wybieramy **AutoGluon CatBoost**. Ma najwyższy recall, a jego czas predykcji, rozmiar i pamięć
są wciąż małe.

| Model | Recall | Predykcja 1 rekordu (ms) | Rozmiar artefaktu | RAM po wczytaniu (MiB) |
|---|---:|---:|---:|---:|
| Regresja logistyczna | 47,19% | 1,816 | 3,9 KB | 0,066 |
| AutoGluon LightGBM | 75,28% | 3,359 | 305 KB | 17,895 |
| AutoGluon CatBoost | 86,52% | 3,956 | 809 KB | 25,250 |

Źródło: [`docs/sprint3_model_comparison.md`](../sprint3_model_comparison.md), runy W&B
z pipeline'u `experiments` (git SHA `91188ea`).

CatBoost wykrywa 77 z 89 odchodzących klientów, LightGBM 67, a regresja logistyczna 42.

Postać modelu:

- pakiet modelu w `data/06_models/production`: katalog `model` (predyktor AutoGluon)
  i `metadata.json` (cechy, target, klasa pozytywna, metryki, git SHA, fingerprint danych,
  wersje bibliotek);
- zapis i odczyt tylko przez wspólne funkcje `save_model_package` i `load_model_package`
  z [`model_package.py`](../../src/asi_project/model_package.py);
- w W&B artefakt `iranian-churn-model` z aliasem `production`;
- [`scripts/smoke_model.py`](../../scripts/smoke_model.py) pobiera artefakt `production`,
  wczytuje go i robi predykcję dla 5 wierszy.

## Odrzucone alternatywy

- **Regresja logistyczna:** jest najszybsza i najmniejsza, ale wykrywa mniej niż połowę
  odchodzących klientów (recall 47,19%).
- **AutoGluon LightGBM:** jest mniejszy i trochę szybszy od CatBoost, ale wykrywa o 10
  odchodzących klientów mniej (recall 75,28%). Różnica ok. 0,6 ms i ok. 7 MiB RAM nie jest dla
  nas ważniejsza niż recall.

## Konsekwencje

- API w Sprincie 4 pobiera model z W&B po aliasie `production` i nie musi znać numeru runu.
- Obraz API będzie potrzebował biblioteki AutoGluon z CatBoost, więc będzie większy niż przy
  regresji logistycznej.
- Jeśli wybierzemy inny model, wystarczy przenieść alias `production` na nowy artefakt
  i zaktualizować ten ADR.

## Historia

| Data | Zmiana |
|---|---|
| 2026-11-24 | utworzenie |
