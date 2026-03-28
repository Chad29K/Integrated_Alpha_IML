# Alpha Stock

This repository contains our IML course project on short-horizon stock ranking
in the A-share market. The main task is to combine a reinforcement-learning
factor search module with an LSTM forecasting module and compare the signals
they produce on the same dataset.

## Project Scope

- `Data` module: loads the local labelled stock panel and basic summary statistics.
- `RL` module: searches over simple symbolic factors with a small Q-learning setup.
- `LSTM` module: predicts future 20-day return from rolling sequences of market features.
- `LSTM price demo`: a single-stock next-price example used only for visual explanation.
- `LLM` module: an optional interface for asking questions about the current outputs.
- `dashboard_app.py`: a local dashboard for browsing the latest results.

The coursework requirement asks for at least two techniques from the given list.
This project mainly uses:

1. Reinforcement Learning
2. LSTM

The optional chat interface is not treated as the core technical contribution.

## Quick Start

1. Open PowerShell in this folder.
2. Create and activate a virtual environment if needed.
3. Install dependencies from `requirements.txt`.
4. If you want to use data refresh or the optional chat interface, copy `.env.example` to `.env` and fill in the required keys.
5. Run the full pipeline:

```powershell
.\.venv\Scripts\python.exe main.py run-all
```

6. If needed, open the optional chat interface after the pipeline finishes:

```powershell
.\.venv\Scripts\python.exe main.py chat
```

## Main Commands

```powershell
.\.venv\Scripts\python.exe main.py data-summary
.\.venv\Scripts\python.exe main.py tushare-sync
.\.venv\Scripts\python.exe main.py rl
.\.venv\Scripts\python.exe main.py lstm
.\.venv\Scripts\python.exe main.py lstm-price-demo --stock-code 000001.SZ
.\.venv\Scripts\python.exe main.py run-all
.\.venv\Scripts\python.exe main.py chat
.\.venv\Scripts\python.exe dashboard_app.py --host 127.0.0.1 --port 8000
```

## Folder Layout

- `data/labeled`: local CSV files used in the experiments.
- `src/integrated_alpha/data_module`: data loading and refresh utilities.
- `src/integrated_alpha/lstm_module`: sequence modelling code.
- `src/integrated_alpha/rl_module`: symbolic factor search and ranking.
- `src/integrated_alpha/llm_module`: optional question-answer interface.
- `src/integrated_alpha/dashboard_module`: helper logic for the dashboard.
- `outputs`: saved metrics, plots, summaries, and dashboard artefacts.
- `tests`: smoke tests for the main pipeline and dashboard flow.

## Dashboard

Run the local dashboard with:

```powershell
.\.venv\Scripts\python.exe dashboard_app.py --host 127.0.0.1 --port 8000
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

The dashboard shows:

- the top ranked stocks for the latest trade date
- a detail page for each selected stock
- RL and LSTM based explanation text
- historical fit charts
- an optional chat box connected to the current outputs

## Tushare Refresh

If the local CSV snapshot needs updating, the dataset can be rebuilt from Tushare:

1. Copy `.env.example` to `.env`
2. Add `TUSHARE_TOKEN=...`
3. Run:

```powershell
.\.venv\Scripts\python.exe main.py tushare-sync
```

The refreshed files will be written to `data/tushare_labeled`. If that folder
contains the full tracked universe, the project will read from it automatically.

## Notes

- The main evaluation target is future 20-day return ranking, not exact price matching.
- The single-stock price demo is included only to make one part of the model output easier to interpret.
- The RL search space is deliberately small, so the method is easy to inspect and rerun in a coursework setting.
Note: large local datasets and generated outputs are not included in this repository.
