defmodule Crucible.SignalRecord do
  @moduledoc """
  V4 canonical extracted signal record.
  """

  alias Crucible.{TensorRef, TensorSummary}

  @derive Jason.Encoder
  defstruct [
    :signal_id,
    :trace_id,
    :run_id,
    :signal_type,
    :provider_kind,
    :model_id,
    :model_family,
    :backend,
    :layer_index,
    :token_index,
    :node_name,
    :capture_method,
    :tensor_summary,
    :tensor_ref,
    metadata: %{}
  ]

  @type t :: %__MODULE__{}

  def from_legacy(%{signal_ref: ref} = record) do
    %__MODULE__{
      signal_id: ref.signal_id,
      trace_id: ref.trace_id,
      run_id: Map.get(ref.metadata, :run_id),
      signal_type: ref.signal_type,
      provider_kind: Map.get(ref.metadata, :provider_kind),
      model_id: ref.model_ref,
      model_family: Map.get(ref.metadata, :model_family),
      backend: Map.get(ref.metadata, :backend),
      layer_index: ref.layer_index,
      token_index: ref.token_index,
      node_name: Map.get(ref.metadata, :node_name),
      capture_method: Map.get(ref.metadata, :capture_method),
      tensor_summary: normalize_summary(Map.get(record, :summary)),
      tensor_ref: normalize_tensor_ref(Map.get(record, :value_ref)),
      metadata: Map.get(record, :metadata, %{})
    }
  end

  defp normalize_summary(%TensorSummary{} = summary), do: summary

  defp normalize_summary(%CrucibleSignal.TensorSummary{} = summary),
    do: TensorSummary.from_legacy(summary)

  defp normalize_summary(nil), do: nil
  defp normalize_summary(summary) when is_map(summary), do: struct(TensorSummary, summary)

  defp normalize_tensor_ref(%TensorRef{} = ref), do: ref
  defp normalize_tensor_ref(nil), do: nil
  defp normalize_tensor_ref(ref) when is_map(ref), do: struct(TensorRef, ref)
end
