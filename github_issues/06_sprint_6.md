# Sprint 6 — Wdrożenie, obserwowalność i release

> Treść generowana z `docs/sprints/SPRINT_6.md` przez `python scripts/build_issues.py`. Zmiany wprowadzaj w pliku sprintu.

**Daty:** 07.01.2027 – 21.01.2027 · **Spotkania:** 07.01 (start sprintu), 14.01 (pełne — release candidate), 21.01 (finał) · **Deadline:** release candidate 14.01.2027 (wewnętrzny, twardy), finał i release v1.0 21.01.2027

## Cel

API i UI działają publicznie w Cloud Run, z tokenem dostępu z Secret Managera, jawnym trybem
persistence, logami i smoke testem. Projekt jest wydany jako v1.0.

## Definition of Done

- [ ] 14.01 (release candidate): API i UI wdrożone; smoke zielony w trybie skonfigurowanym dla chmury — w trybie `disabled` predykcja działa i ma poprawny kontrakt bez zapisu stanu, a w trybie `sql` (przy OPTIONAL bazie w chmurze) rekord jest zapisany.
- [ ] Działanie UI na Cloud Run potwierdzone tym samym manualnym UI checkiem co w Sprincie 5 (podczas release candidate i demo).
- [ ] 21.01: finalny smoke, dokumentacja, release v1.0, demo.
- [ ] `API_KEY` nie jest hardcodowany, nie trafia do repozytorium i nie jest wpisywany ręcznie jako jawna konfiguracja; usługi otrzymują go przez integrację z Secret Managerem.

## Podział pracy (uzupełnia zespół na początku sprintu)

| Element sprintu | Owner | Recenzent |
| --- | --- | --- |
| _do uzupełnienia_ | | |

Integrator sprintu: _do uzupełnienia_

## Pełny opis

Plik `docs/sprints/SPRINT_6.md` w repozytorium.
