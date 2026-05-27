# Tensor Summaries

Tensor summaries provide deterministic bounded statistics for traces and policies.

## What This Covers

Use `PartialSummary` for mergeable streaming statistics. Entropy and quantiles require raw distributions or scalar token-level aggregates.

## Worked Example

```elixir
alias CrucibleSignal.TensorSummary

left = TensorSummary.from_list([1, 2])
right = TensorSummary.from_list([3, 4])
{:ok, merged} = TensorSummary.merge(left, right)
merged.mean
```

## Related Guides

- [Signal Types](signal_types.md)
- [Testing](testing.md)
