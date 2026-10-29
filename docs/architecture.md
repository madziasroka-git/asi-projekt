# Architektura systemu

Ten dokument opisuje architekturę systemu zespołu. W Sprincie 1 powstaje diagram v0, który
będzie aktualizowany w kolejnych sprintach.

## Architektura docelowa (scaffold)

Tak wygląda system po sześciu sprintach — na poziomie warstw. To mapa odpowiedzialności,
a nie diagram do skopiowania: diagram v0 (niżej) rysujecie sami.

Relacje między warstwami (opis tekstowy, nie diagram):

- training: dane → pipeline Kedro → artefakt modelu
- serving: UI → API → model (artefakt wczytany przez API)
- state: API → persistence
- deployment: API + UI → Cloud Run

| Warstwa | Odpowiedzialność | Sprint | Miejsce w repozytorium |
|---|---|---|---|
| Dane i kontrakt | źródło danych, kontrakt, input normalization, training-only filtering | S1 | `data/01_raw/`, `src/asi_project/datasets.yaml`, `src/asi_project/data.py` |
| Pipeline | odtwarzalne przetwarzanie i trening od surowych danych | S2 | `conf/`, `src/asi_project/pipelines/` |
| Eksperymenty i artefakt | porównanie runów, wybór modelu, pakiet modelu ładowalny poza treningiem | S3 | W&B, pakiet modelu, `docs/model_card.md` |
| API | kontrakt usługi, walidacja na granicy, model wczytywany raz | S4 | `src/asi_project/api/`, `docker/` |
| Persistence | zapis predykcji poza procesem aplikacji (tryby `sql` / `disabled`) | S4 → S5 → S6 | repository w API, Postgres w compose |
| UI | formularz jako klient API | S5 | `ui/`, `docker/`, `docker-compose.yml` |
| Chmura | wdrożenie, sekret, logi, release | S6 | `deploy/`, Cloud Run, Secret Manager, Cloud Logging |

## Diagram v0 (Sprint 1)

Narysujcie, jak — według waszej obecnej wiedzy — będzie wyglądał system na koniec projektu.
Diagram v0 nie musi być poprawny w szczegółach; ma pokazać wasze rozumienie przepływu.
Sama lista warstw z sekcji wyżej nie wystarcza — diagram pokazuje wasz system i wasze decyzje.

Na diagramie zaznaczcie co najmniej:

- skąd pochodzą dane i gdzie żyje kontrakt danych;
- gdzie działa każdy z trzech typów przetwarzania: input normalization, training-only
  filtering, model preprocessing;
- gdzie powstaje model i jak trafia tam, gdzie wykonuje się predykcja;
- kto lub co wysyła żądanie predykcji i co dostaje w odpowiedzi;
- gdzie przechowywany jest stan (np. zapis predykcji);
- które elementy już istnieją po Sprincie 1, a które są planowane.

Diagram umieśćcie w bloku `mermaid` w sekcji poniżej (GitHub renderuje Mermaid w Markdown)
albo jako obraz w `docs/` z linkiem w tej sekcji. Usuńcie znaczniki TODO, gdy diagram będzie
gotowy — sprawdza to `python scripts/check_dod.py`.

<!-- DIAGRAM-V0:START -->
```mermaid
flowchart LR
    csv["Dane CSV<br/>Sprint 1 - jest"] --> data["data.py<br/>Sprint 1 - jest"]
    data --> modelpy["model.py<br/>Sprint 1 - jest"]
    modelpy --> kedro["Kedro<br/>Sprint 2"]
    kedro --> model["Model<br/>Sprint 3"]
    model --> api["API<br/>Sprint 4"]
    api --> ui["UI<br/>Sprint 5"]
    api --> db[("Baza danych<br/>Sprint 4/5")]
    api --> cloud
    ui --> cloud(("Cloud Run<br/>Sprint 6"))
```

Dane wejściowe pochodzą z pliku CSV (`data/01_raw/`), a kontrakt danych żyje w
`src/asi_project/datasets.yaml` i jest egzekwowany w `data.py`, gdzie odbywa się input
normalization oraz training-only filtering. `model.py` zawiera model preprocessing i baseline
— te trzy elementy istnieją już po Sprincie 1. W Sprincie 2 przetwarzanie i trening zostaną
opakowane w odtwarzalny pipeline Kedro, a w Sprincie 3 powstanie wybrany artefakt modelu,
który API (Sprint 4) wczyta raz przy starcie. UI (Sprint 5) wysyła żądania predykcji do API
i dostaje w odpowiedzi wynik predykcji, a API zapisuje predykcje w bazie danych (Sprint 4/5).
Na koniec API i UI zostaną wdrożone na Cloud Run (Sprint 6).
<!-- DIAGRAM-V0:END -->

## Historia zmian architektury

| Sprint | Zmiana | Powiązany ADR |
|---|---|---|
| S1 | diagram v0 | — |
