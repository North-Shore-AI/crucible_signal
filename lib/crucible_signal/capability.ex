defmodule CrucibleSignal.Capability do
  @moduledoc """
  Adapter capability facts for a signal type.
  """

  alias CrucibleSignal.{CapabilityStatus, CaptureMode, Operation, SafeTerms, SignalType}

  @derive Jason.Encoder
  defstruct signal_type: nil,
            operations: [],
            capture_modes: [:summary],
            adapter: nil,
            model_family: nil,
            status: :captured,
            reason: nil,
            metadata: %{}

  @type t :: %__MODULE__{}

  def new(attrs) when is_list(attrs) or is_map(attrs) do
    attrs = normalize_attrs(attrs)

    with {:ok, raw_signal_type} <- fetch_required(attrs, :signal_type),
         {:ok, signal_type} <- SignalType.normalize(raw_signal_type),
         {:ok, operations} <- normalize_many(Map.get(attrs, :operations, [:read]), Operation),
         {:ok, capture_modes} <-
           normalize_many(Map.get(attrs, :capture_modes, [:summary]), CaptureMode),
         {:ok, status} <- CapabilityStatus.normalize(Map.get(attrs, :status, :captured)) do
      {:ok,
       struct(__MODULE__, %{
         signal_type: signal_type,
         operations: operations,
         capture_modes: capture_modes,
         adapter: Map.get(attrs, :adapter),
         model_family: Map.get(attrs, :model_family),
         status: status,
         reason: Map.get(attrs, :reason),
         metadata: Map.get(attrs, :metadata, %{})
       })}
    end
  end

  def new!(attrs) do
    case new(attrs) do
      {:ok, capability} -> capability
      {:error, reason} -> raise ArgumentError, "invalid capability: #{inspect(reason)}"
    end
  end

  def supports?(%__MODULE__{} = capability, operation) do
    with {:ok, operation} <- Operation.normalize(operation) do
      operation in capability.operations
    else
      _ -> false
    end
  end

  def supports_capture?(%__MODULE__{} = capability, capture_mode) do
    with {:ok, capture_mode} <- CaptureMode.normalize(capture_mode) do
      capture_mode in capability.capture_modes
    else
      _ -> false
    end
  end

  defp normalize_attrs(attrs), do: SafeTerms.normalize_attrs(attrs)

  defp fetch_required(attrs, field) do
    case Map.fetch(attrs, field) do
      {:ok, value} when value not in [nil, ""] -> {:ok, value}
      _ -> {:error, {:missing_required_fields, [field]}}
    end
  end

  defp normalize_many(values, module) do
    values
    |> List.wrap()
    |> Enum.reduce_while({:ok, []}, fn value, {:ok, acc} ->
      case module.normalize(value) do
        {:ok, value} -> {:cont, {:ok, [value | acc]}}
        {:error, reason} -> {:halt, {:error, reason}}
      end
    end)
    |> case do
      {:ok, values} -> {:ok, Enum.reverse(values)}
      {:error, reason} -> {:error, reason}
    end
  end
end
