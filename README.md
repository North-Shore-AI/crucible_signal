<p align="center">
  <img src="assets/crucible_signal.svg" width="200" height="200" alt="crucible_signal logo" />
</p>

<p align="center">
  <a href="https://github.com/North-Shore-AI/crucible_signal">
    <img alt="GitHub: crucible_signal" src="https://img.shields.io/badge/GitHub-crucible_signal-0b0f14?logo=github" />
  </a>
  <a href="https://github.com/North-Shore-AI/crucible_signal/blob/main/LICENSE">
    <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-0b0f14.svg" />
  </a>
</p>

# CrucibleSignal

Canonical Elixir signal ontology for transformer forward-pass artifacts, tensor
summaries, capabilities, and internal-control surfaces. The package is
model-agnostic; model-specific capture belongs in adapter packages.

## Stack Position

`crucible_signal` is the lowest-level Crucible package in the forward-pass
substrate. It owns reusable signal contracts and avoids Bumblebee, Trinity,
runtime supervision, trace persistence, and policy decisions.

## Installation

```elixir
def deps do
  [
    {:crucible_signal, "~> 0.1.0"}
  ]
end
```

## Boundary

This package defines the vocabulary for embeddings, residuals, attention
artifacts, MLP gates, cache metadata, logits, decoded text, and related
operation capabilities. Adapter-specific capture belongs in `crucible_tap` and
`crucible_bumblebee`.

## Usage

```elixir
alias CrucibleSignal.{SignalRef, TensorSummary}

ref =
  SignalRef.new!(
    trace_id: "trace-1",
    signal_id: "final-logits:0",
    signal_type: :final_logits,
    model_ref: "model:local",
    dtype: :f32,
    shape: {1, 151_936}
  )

summary = TensorSummary.summarize([0.1, 0.4, 0.2], entropy: true)
```

## Mechanistic-Interpretability Metadata

Signals can carry TransformerLens-compatible activation metadata without adding
model-family-specific signal types:

```elixir
ref =
  SignalRef.for_activation("blocks.0.attn.hook_q",
    trace_id: "trace-1",
    signal_id: "q0"
  )

ref.signal_type
#=> :attention_q

ref.metadata
#=> %{
#=>   activation_name: "blocks.0.attn.hook_q",
#=>   component: :attn,
#=>   layer_index: 0,
#=>   axes: [:batch, :pos, :head, :d_head]
#=> }
```

`CrucibleSignal.activation_metadata/1` validates canonical activation names,
axes, layer/head indexes, capture mode, and raw artifact references. Activation
claims with missing or semantically invalid axes fail during construction of
`SignalRef`, `SignalSpec`, `Capability`, and `Crucible.SignalRecord`.

## Guides

- [Quickstart](guides/quickstart.md)
- [Concepts](guides/concepts.md)
- [Signal Types](guides/signal_types.md)
- [Tensor Summaries](guides/tensor_summaries.md)
- [Capabilities And Operations](guides/capabilities_and_operations.md)
- [Working Examples](guides/working_examples.md)
- [Testing](guides/testing.md)

## Examples

- `examples/signal_factory_mock.exs`
- `examples/tensor_summary_live.exs`

## Testing

- Default suite: `mix test`
- Full local gate: `mix ci`

Documentation can be generated with `mix docs` and published to HexDocs.

## V5 Status

Status: `signal-ontology-real-output-passing`.

V5 keeps the provider-neutral `Crucible.*` DTOs and expands the signal
vocabulary used by native Bumblebee and Python/PyTorch traces: input IDs,
attention masks, final logits, generation-step logits, hidden states, attention
weights, residual/MLP summaries, router/MoE probes, KV-cache metadata, backend
events, and capability records.

The V5 gate round-trips real model output summaries through JSON without raw
tensor arrays. Phase artifacts are recorded in the V5 checklist, including
`tmp/crucible_v5/transcripts/crucible_signal_mix_ci.log`.
