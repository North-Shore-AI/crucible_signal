# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- Added V4 provider-neutral `Crucible.*` tensor, artifact, metadata, and signal
  record DTOs plus deterministic canonical JSON digest helpers.
- Added the V4 signal vocabulary used by native Bumblebee traces and replay.
- Added V5 real-output signal coverage for input IDs, attention masks, final
  logits, generation-step logits, hidden states, attention weights,
  residual/MLP summaries, router/MoE probes, KV-cache metadata, backend events,
  and explicit capability status metadata.
- Added model-agnostic signal types for MoE router logits, world-model state,
  verifier signals, logit-lens intermediates, and KV cache state.
- Added factory helpers, compressed-vector capture mode, partial summaries,
  count-compatible merge validation, guides, and runnable examples.
