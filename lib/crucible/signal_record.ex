defmodule Crucible.SignalRecord do
  @moduledoc """
  V4/V5 canonical extracted signal record.
  """

  alias Crucible.{TensorRef, TensorSummary}
  alias CrucibleSignal.ActivationMetadata

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

  @spec new(map() | keyword()) :: {:ok, t()} | {:error, term()}
  def new(attrs) when is_list(attrs), do: attrs |> Map.new() |> new()

  def new(attrs) when is_map(attrs) do
    with {:ok, tensor_summary} <- canonical_tensor_summary(Map.get(attrs, :tensor_summary)),
         {:ok, tensor_ref} <- canonical_tensor_ref(Map.get(attrs, :tensor_ref)),
         {:ok, metadata} <- canonical_metadata(Map.get(attrs, :metadata, %{})) do
      record =
        attrs
        |> Map.put(:tensor_summary, tensor_summary)
        |> Map.put(:tensor_ref, tensor_ref)
        |> Map.put(:metadata, metadata)

      {:ok, struct(__MODULE__, record)}
    end
  end

  @spec new!(map() | keyword()) :: t()
  def new!(attrs) do
    case new(attrs) do
      {:ok, record} -> record
      {:error, reason} -> raise ArgumentError, "invalid signal record: #{inspect(reason)}"
    end
  end

  defp canonical_tensor_summary(nil), do: {:ok, nil}
  defp canonical_tensor_summary(%TensorSummary{} = summary), do: {:ok, summary}

  defp canonical_tensor_summary(summary) when is_map(summary),
    do: {:ok, struct(TensorSummary, summary)}

  defp canonical_tensor_summary(_summary), do: {:error, :invalid_tensor_summary}

  defp canonical_tensor_ref(nil), do: {:ok, nil}
  defp canonical_tensor_ref(%TensorRef{} = ref), do: {:ok, ref}
  defp canonical_tensor_ref(ref) when is_map(ref), do: {:ok, struct(TensorRef, ref)}
  defp canonical_tensor_ref(_ref), do: {:error, :invalid_tensor_ref}

  defp canonical_metadata(metadata), do: ActivationMetadata.normalize(metadata)
end
