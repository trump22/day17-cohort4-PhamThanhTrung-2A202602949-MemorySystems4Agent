# Implementation

- `model_provider.py`: normalized providers and lazy LangChain adapters.
- `config.py`: environment config; offline by default.
- `memory_store.py`: UTF-8 User.md CRUD, structured facts, conservative extraction and incremental compact summaries.
- `agent_baseline.py`: per-thread history without persistent profiles.
- `agent_advanced.py`: profile + summary + recent messages, live/offline responses.
- `benchmark.py`: isolated, deterministic suites with six required metrics.
- `test_agents.py`: 34 behavior tests including corrections, persistence, isolation, prompt savings and mocked live providers.

From repository root: `python -m pytest src/test_agents.py -v` and `python src/benchmark.py --output benchmark_results.json`.
See `REPORT.md` for measured results and limitations.
