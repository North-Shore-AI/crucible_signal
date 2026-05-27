# Quickstart

Create validated signal references and tensor summaries with model-agnostic contracts.

## What This Covers

This guide covers the smallest useful path through `CrucibleSignal`.

## Worked Example

```elixir
alias CrucibleSignal.{SignalRef, TensorSummary}

ref = SignalRef.for_final_logits(trace_id: "trace-1", model_ref: "model:fixture")
summary = TensorSummary.from_list([0.1, 0.7, 0.2], entropy: true)

{ref.signal_type, summary.entropy}
```

## Related Guides

- [Concepts](concepts.md)
- [Signal Types](signal_types.md)
- [Tensor Summaries](tensor_summaries.md)
