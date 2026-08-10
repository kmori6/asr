# Repository instructions

## Purpose

This file preserves durable engineering decisions for research-oriented deep-learning repositories. It is intended to
be reusable when starting related projects such as machine translation, ASR, TTS, or speaker recognition.

When copying this file to another repository, keep the general rules and update the repository-specific context at the
end. Source code, tests, configuration files, and the README remain the source of truth for concrete behavior; do not
use this file as a snapshot of the file tree or as a substitute for documentation.

## Engineering priorities

Make decisions in this order:

1. Correctness and explicit semantics.
2. Readability and simplicity.
3. Reproducibility and maintainability.
4. Extensibility supported by a real use case.
5. Performance demonstrated by profiling or a clear tensor-level cost.

Prefer conventional, unsurprising code over clever code. Preserve the intent of a referenced paper or established
implementation, and document deliberate deviations.

## Repository organization

Use a small three-part structure:

```text
src/<package>/                 # reusable library code
experiments/<experiment>/     # experiment-specific orchestration
tests/                         # unit and integration tests
```

Directories below `src/<package>/` are capability-based and optional. Create only the packages required by the current
domain. Possible examples include:

- `data/` for reusable datasets, collators, transforms, and feature processing.
- `models/` for complete end-to-end model architectures.
- `modules/` for reusable neural-network components used to build models.
- `training/` for reusable training loops, schedulers, checkpointing, and metric history.
- `decoding/` for explicit decoding or search algorithms such as greedy or beam search.
- `generation/` or `synthesis/` for a distinct generation pipeline when that terminology better matches the domain.
- `evaluation/` for reusable evaluation logic that is substantial enough to live outside experiment scripts.

Do not reproduce this list mechanically. A classification repository may have no `decoding/` package, and a TTS
repository may use `synthesis/` instead. Do not create empty packages or generic layers merely to make different
repositories look identical. Choose names from the responsibility the code actually has.

Keep dependency direction clear:

- Experiment scripts may import reusable code from `src/<package>/`; reusable code must not import experiment code.
- Models may compose modules. Modules must not depend on a complete model.
- Scripts orchestrate configuration, data, models, training, and evaluation. Do not implement reusable model or
  algorithm logic in scripts.
- Add a new package only when its contents form a coherent responsibility that can evolve independently.

Each experiment should be understandable from its own directory. A typical layout is:

```text
experiments/<experiment>/
├── config/
├── scripts/
├── data/       # local or generated; not committed
└── results/    # generated artifacts; not committed
```

The exact subdirectories are optional. Commit the scripts and configuration required to reproduce an experiment, and
ignore downloaded data, checkpoints, plots, logs, and other generated artifacts at the repository root.

## Design principles

Use SOLID as a set of design heuristics, not as a goal by itself. Prefer KISS and YAGNI when there is no demonstrated
variation. Do not add an interface, base class, factory, wrapper, or file solely for hypothetical reuse. Prefer
composition over inheritance.

- Single Responsibility: give a module, class, or function one clear reason to change. Separate model computation,
  data processing, training orchestration, evaluation, and artifact output when they evolve independently.
- Open/Closed: add an extension point only when multiple implementations exist or near-term variation is explicitly
  required. Prefer a direct implementation before that point.
- Liskov Substitution: implementations sharing an interface must preserve documented behavior, including tensor
  shapes, dtypes, devices, mask semantics, state, and cache behavior.
- Interface Segregation: keep `Protocol` definitions and other interfaces minimal. A consumer should depend only on
  operations it actually calls.
- Dependency Inversion: high-level algorithms such as training, decoding, or generation may depend on a minimal
  protocol when multiple models or test doubles must be supported. Use a concrete type when concrete semantics are
  required; do not widen it to `nn.Module` merely to appear generic.

If these principles conflict with framework conventions, clarity, or measured performance, explain the trade-off and
choose the simplest design that preserves correctness.

## APIs, names, and documentation

- Use domain-specific names that reveal intent. Avoid catch-all modules such as `utils.py`, `common.py`, or `types.py`
  when a more precise name is available.
- Keep the public API small. Prefix implementation-only helpers with `_`, and re-export only intentionally supported
  names from `__init__.py`.
- Keep a result dataclass, a small protocol, or a private helper in the consuming module until it has a second genuine
  consumer or an independent responsibility.
- Use current Python type syntax and precise tensor/container types. Avoid `Any` when the contract can be expressed.
- Document non-obvious public behavior. Tensor APIs should state relevant shapes, dtype or mask conventions, device
  expectations, return values, and stateful behavior such as incremental inference.
- Comments should explain reasons, invariants, numerical choices, and deviations rather than restating code.
- For non-obvious library-specific behavior or paper-derived settings, link to a primary source in a nearby comment
  when that source materially helps maintain the implementation.

## PyTorch implementation

- Prefer clear PyTorch-native tensor operations over manual element-wise Python loops.
- Vectorize work across batch, token/frame, class, and beam dimensions when practical. A Python loop is acceptable for
  a naturally sequential or small structural dimension such as a stack of layers.
- Use optimized framework operations when they preserve the required semantics and improve clarity or measured
  performance. Do not sacrifice architectural requirements merely to fuse operations.
- Make shapes and transformations easy to follow. Use semantic variable names rather than numbered intermediates.
- Preserve device and dtype unless an explicit conversion is required. Create new tensors and masks on the appropriate
  device.
- Validate configuration and structural invariants early, especially dimensions, head divisibility, sequence limits,
  vocabulary and special-token IDs, and cache shapes.
- Keep training and inference behavior explicit. Use `torch.inference_mode()` for evaluation and device-appropriate
  mixed precision only where supported.
- Treat `torch.compile` as an execution optimization, not part of model semantics. Save and load the underlying model
  state in a way that also works without compilation.

For an autoregressive model, introduce incremental state only when inference needs it. Cache the actual reusable state
(for example projected keys and values), define who owns and reorders it, and test incremental output against full
causal output. Static encoder-side state and growing decoder-side state should have distinct, explicit semantics.

## Data and experiment configuration

- Keep preprocessing deterministic where possible and make train/evaluation transformations consistent.
- Store normalization and special-token behavior with the tokenizer or processor when the library supports it, rather
  than duplicating that behavior across scripts.
- Validate external artifacts such as tokenizers and checkpoints before a long run. Fail early on incompatible sizes,
  IDs, shapes, or configuration.
- Put reproducible choices in versioned configuration: architecture, optimizer, scheduler, seed, paths, and decoding or
  generation settings.
- Keep machine-specific paths and generated output locations out of reusable library code. Resolve experiment-relative
  paths consistently.
- Avoid hidden working-directory changes and automatically created run-directory trees unless the experiment explicitly
  needs them.

## Training and evaluation contracts

- A model used by the shared trainer should return a mapping of scalar tensor metrics containing `loss`. The trainer
  may treat `loss` specially for backpropagation and model selection, but should aggregate other scalar metric names
  generically.
- Unless exact weighting is explicitly required, epoch metrics may be unweighted means of per-batch metrics. Keep a
  NOTE near that implementation explaining that variable-length batches make this an approximation rather than an
  exact token- or frame-level mean.
- Checkpoints needed for resumption should include model, optimizer, scheduler, scaler, epoch, and selection state.
- Keep history and visualization generic: persist epoch records and produce plots from available scalar metric keys
  instead of hard-coding task-specific metric names.
- Evaluation must use the same processor/tokenizer contract and compatible model configuration as training.
- Use established task metrics where possible and save enough metadata to reproduce them. Save machine-readable
  metrics and useful predictions/references in the experiment result directory.

## Testing

- Organize tests by the source responsibility they cover. Mirroring the `src/` tree is useful but not mandatory when a
  different grouping is clearer.
- Do not add `__init__.py` under `tests/` unless package semantics are specifically required.
- Add the smallest meaningful test for each behavior. Test the way production or experiment code actually calls it.
- For tensor modules, verify the relevant shape and finite values, plus the behavior most likely to regress: masks,
  gradients, padding, state transitions, cache equivalence, serialization, or error handling.
- A bug fix requires a regression test that fails for the original bug.
- Keep unit tests deterministic, fast, and CPU-compatible unless the behavior is specifically device-dependent.
- Do not duplicate implementation details in tests. Assert public behavior and important invariants.

## Dependencies and tooling

- Use `uv` for environment and dependency management. Put development-only tools in the development dependency group
  and commit `uv.lock` for reproducibility.
- Treat `pyproject.toml` as the source of truth for formatter, linter, type-checker, and test configuration.
- Avoid a new dependency when the standard library or an existing dependency provides a clear implementation. When a
  dependency is justified, add it with `uv` instead of editing only the lockfile.

After changing Python code, run the narrowest relevant tests first, then run the applicable repository checks:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy --ignore-missing-imports .
uv run pytest
```

Apply formatting with `uv run ruff format .` when needed. Report any check that could not be run and why.

## Change workflow

- Inspect the implementation, call sites, tests, configuration, and current working-tree changes before editing.
- Preserve user changes and avoid unrelated rewrites.
- For a review, diagnosis, explanation, or design request, report evidence and trade-offs without implementing changes
  unless implementation is also requested.
- For an implementation or fix request, update the code and its directly affected tests, imports, configuration, and
  documentation, then perform relevant non-destructive validation.
- Prefer a focused patch over a broad refactor. If a broader change is necessary, explain the dependency that makes it
  necessary.
- Do not claim that a change works without running an appropriate check, and do not conceal failing checks that appear
  related to the change.

## Repository-specific context

Update or replace this section when copying the file to another repository.

- Package: `asr`
- Primary task: automatic speech recognition
- Current reference experiment: LibriSpeech under `experiments/librispeech/`
- Reusable implementation: `src/asr/`
- Generated experiment data and results: `experiments/*/data/` and `experiments/*/results/` (not committed)
- Primary environment and command runner: Python 3.12 and `uv`
