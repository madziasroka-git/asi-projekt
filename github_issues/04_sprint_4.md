# Sprint 4 — Usługa, stan i pierwszy kontener

> Treść generowana z `docs/sprints/SPRINT_4.md` przez `python scripts/build_issues.py`. Zmiany wprowadzaj w pliku sprintu.

**Daty:** 26.11.2026 – 17.12.2026 · **Spotkania:** 26.11 (start sprintu), 03.12 i 10.12 (pełne), 17.12 (zamknięcie sprintu i start Sprintu 5) · **Deadline:** stan gałęzi `main` na początku zajęć 17.12.2026

## Cel

Model jest dostępny przez usługę z walidowanym kontraktem i prostym tokenem dostępu, zapisuje
predykcje przez konfigurowalne persistence i działa w kontenerze.

## Definition of Done

- [ ] `docker run` z `.env` → smoke zielony w trybie `sql`: `/healthz` raportuje `sql`, `/predict` z tokenem zwraca poprawny kontrakt, rekord jest zapisany w SQLite na wolumenie.
- [ ] `/predict` bez tokenu zwraca 401.
- [ ] Testy API przechodzą w CI.
- [ ] W repozytorium nie ma sekretów.
- [ ] Pre-flight GCP zespołu jest wykonany.
- [ ] Szkic ADR-3 jest zmergowany.

## Podział pracy (uzupełnia zespół na początku sprintu)

| Element sprintu | Owner | Recenzent |
| --- | --- | --- |
| _do uzupełnienia_ | | |

Integrator sprintu: _do uzupełnienia_

## Pełny opis

Plik `docs/sprints/SPRINT_4.md` w repozytorium.
