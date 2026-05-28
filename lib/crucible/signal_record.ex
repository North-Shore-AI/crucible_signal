defmodule Crucible.SignalRecord do
  @moduledoc """
  V4/V5 canonical extracted signal record.
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
    :model_revision,
    :backend,
    :dtype,
    :shape,
    :rank,
    :device,
    :layer_index,
    :token_index,
    :node_name,
    :capture_method,
    :surface_id,
    :tap_id,
    :capability_status,
    :capability_reason,
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
      model_revision: Map.get(ref.metadata, :model_revision),
      backend: Map.get(ref.metadata, :backend),
      dtype: ref.dtype,
      shape: shape_dims(ref.shape),
      rank: shape_rank(ref.shape),
      device: Map.get(ref.metadata, :device),
      layer_index: ref.layer_index,
      token_index: ref.token_index,
      node_name: Map.get(ref.metadata, :node_name),
      capture_method: Map.get(ref.metadata, :capture_method),
      surface_id: Map.get(ref.metadata, :surface_id),
      tap_id: Map.get(ref.metadata, :tap_id),
      capability_status: Map.get(ref.metadata, :capability_status),
      capability_reason: Map.get(ref.metadata, :capability_reason),
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

  defp shape_dims(%CrucibleSignal.TensorShape{dims: dims}), do: dims
  defp shape_dims(_shape), do: nil

  defp shape_rank(%CrucibleSignal.TensorShape{rank: rank}), do: rank
  defp shape_rank(_shape), do: nil
end
