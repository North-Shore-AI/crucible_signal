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
    :activation_name,
    :component,
    :axes,
    :layer_index,
    :node_name,
    :token_index,
    :head_index,
    :kv_head_index,
    :capture_mode,
    :raw_ref,
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
