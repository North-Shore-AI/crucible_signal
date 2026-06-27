defmodule CrucibleSignal.SafeTerms do
  @moduledoc false

  @key_map %{
    "adapter" => :adapter,
    "bounds" => :bounds,
    "capture_mode" => :capture_mode,
    "capture_modes" => :capture_modes,
    "decode_step" => :decode_step,
    "dtype" => :dtype,
    "head_index" => :head_index,
    "heads" => :heads,
    "id" => :id,
    "layer_index" => :layer_index,
    "layers" => :layers,
    "metadata" => :metadata,
    "model_family" => :model_family,
    "model_ref" => :model_ref,
    "operations" => :operations,
    "reason" => :reason,
    "redaction" => :redaction,
    "required?" => :required?,
    "shape" => :shape,
    "signal_id" => :signal_id,
    "signal_type" => :signal_type,
    "status" => :status,
    "storage_ref" => :storage_ref,
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
