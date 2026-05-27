# Signal Types

Signal types cover model internals while remaining capability-gated per backend.

## What This Covers

The ontology includes dense-model signals, optional MoE signals, verifier/world-model placeholders, and cache state.

## Worked Example

```elixir
CrucibleSignal.SignalType.normalize("logit-lens-intermediate")
CrucibleSignal.SignalType.normalize(:kv_cache)
```

## Related Guides

- [Quickstart](quickstart.md)
- [Capabilities And Operations](capabilities_and_operations.md)
