defmodule CrucibleSignal do
  @moduledoc """
  Canonical signal ontology for transformer forward-pass artifacts.

  This package owns reusable contracts for signal names, references,
  capabilities, tensor metadata, and capture posture. It intentionally avoids
  model loading, orchestration, and product-specific routing.
  """

  @version Mix.Project.config()[:version]

  @doc "Returns the package version."
  def version, do: @version

  @doc "Returns all canonical signal types."
  def signal_types, do: CrucibleSignal.SignalType.all()

  @doc "Returns all canonical operation types."
  def operations, do: CrucibleSignal.Operation.all()

  @doc "Returns all supported capture modes."
  def capture_modes, do: CrucibleSignal.CaptureMode.all()

  defdelegate for_final_logits(attrs \\ []), to: CrucibleSignal.SignalRef
  defdelegate for_layer_residual(layer_index, attrs \\ []), to: CrucibleSignal.SignalRef

  defdelegate for_attention_map(layer_index, head_index, attrs \\ []),
    to: CrucibleSignal.SignalRef

  defdelegate for_mlp_gate(layer_index, attrs \\ []), to: CrucibleSignal.SignalRef
  defdelegate for_norm_telemetry(layer_index, attrs \\ []), to: CrucibleSignal.SignalRef
  defdelegate for_moe_router(layer_index, attrs \\ []), to: CrucibleSignal.SignalRef
  defdelegate for_world_model(attrs \\ []), to: CrucibleSignal.SignalRef
  defdelegate for_verifier(attrs \\ []), to: CrucibleSignal.SignalRef
  defdelegate for_logit_lens(layer_index, attrs \\ []), to: CrucibleSignal.SignalRef
  defdelegate for_kv_cache(decode_step, attrs \\ []), to: CrucibleSignal.SignalRef
  defdelegate for_decoded_text(attrs \\ []), to: CrucibleSignal.SignalRef
end
