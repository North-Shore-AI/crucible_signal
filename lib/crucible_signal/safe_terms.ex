defmodule CrucibleSignal.SafeTerms do
  @moduledoc false

  @key_map %{
    "adapter" => :adapter,
    "activation_name" => :activation_name,
    "axes" => :axes,
    "bounds" => :bounds,
    "capture_mode" => :capture_mode,
    "capture_modes" => :capture_modes,
    "component" => :component,
    "decode_step" => :decode_step,
    "dtype" => :dtype,
    "head" => :head,
    "head_index" => :head_index,
    "kv_head_index" => :kv_head_index,
    "heads" => :heads,
    "id" => :id,
    "intervention_allowed?" => :intervention_allowed?,
    "kind" => :kind,
    "layer" => :layer,
    "layer_index" => :layer_index,
    "layer_name" => :layer_name,
    "layers" => :layers,
    "metadata" => :metadata,
    "model_family" => :model_family,
    "model_ref" => :model_ref,
    "nodes" => :nodes,
    "operations" => :operations,
    "raw_ref" => :raw_ref,
    "reason" => :reason,
    "redaction" => :redaction,
    "required?" => :required?,
    "requires_raw?" => :requires_raw?,
    "selector" => :selector,
    "shape" => :shape,
    "signal_id" => :signal_id,
    "signal_type" => :signal_type,
    "source_signal_id" => :source_signal_id,
    "source_trace_id" => :source_trace_id,
    "status" => :status,
    "storage_ref" => :storage_ref,
    "surface_node_id" => :surface_node_id,
    "token" => :token,
    "token_index" => :token_index,
    "token_range" => :token_range,
    "tokens" => :tokens,
    "trace_id" => :trace_id
  }

  def normalize_attrs(attrs) when is_list(attrs), do: attrs |> Map.new() |> normalize_attrs()

  def normalize_attrs(attrs) when is_map(attrs) do
    Map.new(attrs, fn
      {key, value} when is_binary(key) -> {Map.get(@key_map, key, key), normalize_nested(value)}
      {key, value} -> {key, normalize_nested(value)}
    end)
  end

  def normalize_nested(value) when is_struct(value), do: value
  def normalize_nested(value) when is_map(value), do: normalize_attrs(value)
  def normalize_nested(value) when is_list(value), do: Enum.map(value, &normalize_nested/1)
  def normalize_nested(value), do: value
end
