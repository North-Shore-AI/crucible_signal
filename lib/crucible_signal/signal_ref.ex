defmodule CrucibleSignal.SignalRef do
  @moduledoc """
  Stable reference to a captured or derived forward-pass signal.
  """

  alias CrucibleSignal.{CaptureMode, DType, Redaction, SignalType, TensorShape}

  @derive Jason.Encoder
  defstruct trace_id: nil,
            signal_id: nil,
            signal_type: nil,
            model_ref: nil,
            layer_index: nil,
            token_index: nil,
            token_range: nil,
            head_index: nil,
            decode_step: nil,
            dtype: nil,
            shape: nil,
            capture_mode: :summary,
            storage_ref: nil,
            redaction: :summary_only,
            metadata: %{}

  @type t :: %__MODULE__{}

  @required [:trace_id, :signal_id, :signal_type]

  def new(attrs) when is_list(attrs) or is_map(attrs) do
    attrs = normalize_attrs(attrs)

    with :ok <- require_fields(attrs, @required),
         {:ok, signal_type} <- SignalType.normalize(Map.fetch!(attrs, :signal_type)),
         {:ok, capture_mode} <- CaptureMode.normalize(Map.get(attrs, :capture_mode, :summary)),
         {:ok, redaction} <- Redaction.normalize(Map.get(attrs, :redaction, :summary_only)),
         {:ok, dtype} <- DType.normalize(Map.get(attrs, :dtype)),
         {:ok, shape} <- normalize_shape(Map.get(attrs, :shape)) do
      {:ok,
       struct(__MODULE__, %{
         trace_id: Map.fetch!(attrs, :trace_id),
         signal_id: Map.fetch!(attrs, :signal_id),
         signal_type: signal_type,
         model_ref: Map.get(attrs, :model_ref),
         layer_index: Map.get(attrs, :layer_index),
         token_index: Map.get(attrs, :token_index),
         token_range: Map.get(attrs, :token_range),
         head_index: Map.get(attrs, :head_index),
         decode_step: Map.get(attrs, :decode_step),
         dtype: dtype,
         shape: shape,
         capture_mode: capture_mode,
         storage_ref: Map.get(attrs, :storage_ref),
         redaction: redaction,
         metadata: Map.get(attrs, :metadata, %{})
       })}
    end
  end

  def new!(attrs) do
    case new(attrs) do
      {:ok, ref} -> ref
      {:error, reason} -> raise ArgumentError, "invalid signal ref: #{inspect(reason)}"
    end
  end

  defp normalize_attrs(attrs) when is_list(attrs), do: attrs |> Map.new() |> normalize_attrs()

  defp normalize_attrs(attrs) when is_map(attrs) do
    Map.new(attrs, fn
      {key, value} when is_binary(key) -> {String.to_atom(key), value}
      {key, value} -> {key, value}
    end)
  end

  defp require_fields(attrs, fields) do
    missing = Enum.reject(fields, &present?(attrs, &1))
    if missing == [], do: :ok, else: {:error, {:missing_required_fields, missing}}
  end

  defp present?(attrs, field),
    do: Map.has_key?(attrs, field) and Map.get(attrs, field) not in [nil, ""]

  defp normalize_shape(nil), do: {:ok, nil}
  defp normalize_shape(%TensorShape{} = shape), do: {:ok, shape}

  defp normalize_shape(shape) when is_tuple(shape) or is_list(shape),
    do: {:ok, TensorShape.new!(shape)}

  defp normalize_shape(shape), do: {:error, {:invalid_shape, shape}}
end
