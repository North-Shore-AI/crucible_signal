# Capabilities And Operations

Capabilities state which operations a model adapter can perform for each signal.

## What This Covers

Operation input contracts are data contracts and do not depend on tap structures.

## Worked Example

```elixir
CrucibleSignal.Operation.required_inputs(:route_on)
CrucibleSignal.Operation.describe("steer-model")
```

## Related Guides

- [Concepts](concepts.md)
- [Working Examples](working_examples.md)
