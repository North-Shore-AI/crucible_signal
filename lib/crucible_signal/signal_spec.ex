defmodule CrucibleSignal.SignalSpec do
  @moduledoc """
  Requested signal shape used by tap plans and adapters.
  """

  alias CrucibleSignal.{CaptureMode, Operation, SignalType}

  @derive Jason.Encoder
  defstruct id: nil,
            signal_type: nil,
            layers: :all,
            tokens: :all,
            heads: :all,
            operations: [:read],
            capture_mode: :summary,
            bounds: %{},
            required?: true,
            metadata: %{}

  @type t :: %__MODULE__{}

  def new(attrs) when is_list(attrs) or is_map(attrs) do
    attrs = normalize_attrs(attrs)

    with {:ok, raw_signal_type} <- fetch_required(attrs, :signal_type),
         {:ok, signal_type} <- SignalType.normalize(raw_signal_type),
         {:ok, capture_mode} <- CaptureMode.normalize(Map.get(attrs, :capture_mode, :summary)),
         {:ok, operations} <- normalize_operations(Map.get(attrs, :operations, [:read])) do
      {:ok,
       struct(__MODULE__, %{
         id: Map.get(attrs, :id, "#{signal_type}:#{System.unique_integer([:positive])}"),
         signal_type: signal_type,
         layers: Map.get(attrs, :layers, :all),
         tokens: Map.get(attrs, :tokens, :all),
         heads: Map.get(attrs, :heads, :all),
         operations: operations,
         capture_mode: capture_mode,
         bounds: Map.get(attrs, :bounds, %{}),
         required?: Map.get(attrs, :required?, true),
         metadata: Map.get(attrs, :metadata, %{})
       })}
    end
  end

  def new!(attrs) do
    case new(attrs) do
      {:ok, spec} -> spec
      {:error, reason} -> raise ArgumentError, "invalid signal spec: #{inspect(reason)}"
    end
  end

  defp normalize_attrs(attrs) when is_list(attrs), do: attrs |> Map.new() |> normalize_attrs()

  defp normalize_attrs(attrs) when is_map(attrs) do
    Map.new(attrs, fn
      {key, value} when is_binary(key) -> {String.to_atom(key), value}
      {key, value} -> {key, value}
    end)
  end

  defp fetch_required(attrs, field) do
    case Map.fetch(attrs, field) do
      {:ok, value} when value not in [nil, ""] -> {:ok, value}
      _ -> {:error, {:missing_required_fields, [field]}}
    end
  end

  defp normalize_operations(operations) do
    operations
    |> List.wrap()
    |> Enum.reduce_while({:ok, []}, fn operation, {:ok, acc} ->
      case Operation.normalize(operation) do
        {:ok, operation} -> {:cont, {:ok, [operation | acc]}}
        {:error, reason} -> {:halt, {:error, reason}}
      end
    end)
    |> case do
      {:ok, operations} -> {:ok, Enum.reverse(operations)}
      {:error, reason} -> {:error, reason}
    end
  end
end
