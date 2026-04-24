# Contributing to agentverify

Thanks for your interest in contributing. This document captures the
conventions the project follows so your pull request lands cleanly.

## Development setup

```bash
git clone https://github.com/simukappu/agentverify.git
cd agentverify
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]"
pytest
```

## Commit message conventions

agentverify follows [Conventional Commits](https://www.conventionalcommits.org/).
Scopes are used **only for examples**; every other change uses a bare type.

### Types

| Type | Use for |
|---|---|
| `feat:` | New library features or APIs |
| `fix:` | Bug fixes in the library |
| `refactor:` | Code restructuring without behaviour change |
| `perf:` | Performance improvements |
| `docs:` | Root `README.md`, `CHANGELOG.md`, `.kiro/` contents, docstrings, comments |
| `docs(examples):` | Anything under `examples/` — new example, example README, example tests, example cassette updates |
| `test:` | Test-only changes to the main `tests/` suite |
| `ci:` | CI / GitHub Actions configuration |
| `build:` | Build system, packaging, dependency declarations |
| `chore:` | Maintenance that doesn't fit elsewhere |

### Scope rules

- **`docs(examples):`** is the only scoped type. Use it for every change whose net effect is confined to the `examples/` tree — including example source, its dedicated tests, its README, and its cassette files.
- If a change touches `examples/` **and** the library (`agentverify/` package), use the library-side type (`feat:`, `fix:`, `refactor:`). "Library change that also updates an example" is not an examples change.
- Do not invent other scopes. Any other area of the repo uses a bare type.

### Examples

```
feat: add MATCHES(pattern) regex matcher for string arguments
fix: strip openai Omit/NotGiven sentinels from cassette requests
docs(examples): add LangGraph multi-agent supervisor example
docs: rework README around step-level and data-flow messaging
ci: skip Coveralls upload on forks
```

## CHANGELOG conventions

agentverify keeps a single `CHANGELOG.md` at the repo root. Every
user-facing change is recorded there under an `## Unreleased` section
that gets renamed to `## X.Y.Z (YYYY-MM-DD)` on release.

### Section order

Within a release section, use only the sections below, in this order.
**Omit any section that has no entries in a given release** — don't
leave empty headings.

1. **Features** — new APIs or capabilities added in this release.
2. **Improvements** — behaviour or quality changes to features that already shipped in an earlier release.
3. **Bug Fixes** — fixes for bugs that existed in a previous release.
4. **Dependency** — changes to runtime or optional dependency version requirements.
5. **Deprecated** — APIs marked for removal in a future release.
6. **Breaking Changes** — backwards-incompatible changes. Always a separate section so readers can't miss it.

### Entry rules

- **One sentence per entry, under ~180 characters.** Readers skim; don't write implementation notes.
- **Start each entry with a bolded subject** — a feature name, an API name, or an area of the codebase — so the entries align visually.
- **Describe user-visible behaviour**, not implementation details. Internal helpers, private method names, and line-by-line reasoning belong in the commit body, not the CHANGELOG.
- **Link to README sections or examples** for details when an entry would otherwise grow long.
- **Merge closely related entries.** Two small internal fixes that together improve the same user-visible behaviour should be one entry, not two.
- **Features vs Improvements triage in an Unreleased cycle**: internal tuning of a *new* feature that also lands in this release is folded into the **Features** entry for that feature — it's not a separate Improvement. `Improvements` is reserved for changes to features that users have already seen in a previous release.

### Example (well-formed Unreleased)

```markdown
## Unreleased

### Features

- **Step-level assertions** for agents that make multiple LLM calls per run. `assert_step`, `assert_step_output`, and `assert_step_uses_result_from` verify tool calls, intermediate outputs, and step-to-step data flow on the new `Step` data model. See README "Step-Level Assertions".
- **`MATCHES(pattern)` regex matcher** for verifying string tool-call arguments against a regex, with the same semantics as `ANY`.

### Improvements

- **OpenAI cassette adapter** now also intercepts `AsyncCompletions.create`, so agent frameworks that drive the SDK through `AsyncOpenAI` internally are recorded and replayed transparently.

### Breaking Changes

- **`ExecutionResult` is now step-centric.** `steps: list[Step]` is the single source of truth; `result.tool_calls` is a derived read-only property. All existing read-side code works unchanged. `result.to_dict()` now emits `steps: [...]` instead of `tool_calls: [...]`.
```

## Tests and coverage

- The main test suite runs with `pytest tests/ --cov=agentverify --cov-branch -p no:agentverify`.
- Coverage is gated at **100% line + branch** for the `agentverify/` package. CI fails any drop below.
- Example suites run separately: `pytest examples/<name>/tests`. Smoke checks for the `examples/` tree live in `examples/tests/`.
- When you change the library, add or update tests in `tests/`. When you add an example, add smoke-test entries in `examples/tests/test_smoke.py`.

## Examples

- Every example under `examples/` ships with pre-recorded cassettes so `pytest examples/<name>/tests` runs with no API keys.
- Follow the shape of an existing example when adding a new one: `agent.py` + `tests/conftest.py` + `tests/test_*.py` + `tests/cassettes/*.yaml` + `README.md` + `pyproject.toml`.
- The example's own `README.md` carries the detailed setup / re-record instructions and a table of the tests it ships with. The root README's `## Examples` table gets a one-line entry pointing at the example directory.

## Reporting bugs and requesting features

Use [GitHub Issues](https://github.com/simukappu/agentverify/issues). For bug reports, include a minimal reproduction that can run inside cassette replay mode — it's the fastest path to a fix.

## License

agentverify is [MIT licensed](LICENSE). By contributing, you agree that your contribution will be released under the same terms.
