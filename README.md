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
summaries, capabilities, and internal-control surfaces.

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

Documentation can be generated with `mix docs` and published to HexDocs.
