defmodule Crucible.Signal.Metadata do
  @moduledoc """
  V4/V5 canonical signal provenance metadata.
  """

  @derive Jason.Encoder
  defstruct [
    :model_id,
    :model_family,
    :model_revision,
    :provider_kind,
    :backend,
    :backend_name,
    :dtype,
    :shape,
    :rank,
    :device,
    :layer_index,
    :node_name,
    :token_index,
    :sequence_length,
    :batch_size,
    :capture_method,
    :capability_status,
    :capability_reason,
    :surface_id,
    :tap_id,
    :trace_id,
    :run_id
  ]

  @type t :: %__MODULE__{}
end
