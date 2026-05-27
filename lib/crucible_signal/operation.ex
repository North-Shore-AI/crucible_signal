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
end
