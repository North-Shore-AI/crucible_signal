defmodule CrucibleSignal.Operation do
  @moduledoc """
  Operations a signal can support.
  """

  @operations [
    :read,
    :probe,
    :route_on,
    :fuse,
    :gate,
    :uncertainty,
    :control_vector,
    :shared_memory,
    :verifier_signal,
    :steer_model
  ]

  @type t :: unquote(Enum.reduce(@operations, &{:|, [], [&1, &2]}))

  def all, do: @operations

  def valid?(operation), do: operation in @operations

  def normalize(operation) when is_atom(operation) do
    if valid?(operation), do: {:ok, operation}, else: {:error, {:unknown_operation, operation}}
  end

  def normalize(operation) when is_binary(operation) do
    operation
    |> String.replace("-", "_")
    |> String.to_existing_atom()
    |> normalize()
  rescue
    ArgumentError -> {:error, {:unknown_operation, operation}}
  end

  def normalize(operation), do: {:error, {:unknown_operation, operation}}

  def describe(operation) do
    with {:ok, operation} <- normalize(operation) do
      {:ok, %{operation: operation, required_inputs: required_inputs!(operation)}}
    end
  end

  def required_inputs(operation) do
    with {:ok, operation} <- normalize(operation), do: {:ok, required_inputs!(operation)}
  end

  defp required_inputs!(operation) when operation in [:read, :probe, :uncertainty] do
    [:signal_type, :shape, :capture_mode]
  end

  defp required_inputs!(:route_on), do: [:signal_type, :summary_or_vector]
  defp required_inputs!(:fuse), do: [:logits_or_energy_vector]
  defp required_inputs!(:gate), do: [:signal_type, :gate_condition]
  defp required_inputs!(:control_vector), do: [:vector, :target_signal]
  defp required_inputs!(:shared_memory), do: [:memory_ref, :payload]
  defp required_inputs!(:verifier_signal), do: [:verifier_signal]
  defp required_inputs!(:steer_model), do: [:steering_plan]
end
