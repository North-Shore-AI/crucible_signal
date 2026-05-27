defmodule Crucible.CanonicalJSON do
  @moduledoc """
  Deterministic JSON encoding used for V4 digest inputs.
  """

  def encode!(value) do
    value
    |> canonicalize()
    |> Jason.encode!()
  end

  def digest(value) do
    "sha256:" <> Base.encode16(:crypto.hash(:sha256, encode!(value)), case: :lower)
  end

  def canonicalize(%DateTime{} = value), do: DateTime.to_iso8601(value)

  def canonicalize(value) when is_struct(value) do
    value
    |> Map.from_struct()
    |> canonicalize()
  end

  def canonicalize(value) when is_map(value) do
    value
    |> Enum.map(fn {key, value} -> {key_to_string(key), canonicalize(value)} end)
    |> Enum.sort_by(fn {key, _value} -> key end)
    |> Map.new()
  end

  def canonicalize(value) when is_list(value), do: Enum.map(value, &canonicalize/1)
  def canonicalize(value) when is_tuple(value), do: value |> Tuple.to_list() |> canonicalize()
  def canonicalize(nil), do: nil
  def canonicalize(value) when is_boolean(value), do: value
  def canonicalize(value) when is_atom(value), do: Atom.to_string(value)
  def canonicalize(value), do: value

  defp key_to_string(key) when is_atom(key), do: Atom.to_string(key)
  defp key_to_string(key) when is_binary(key), do: key
  defp key_to_string(key), do: to_string(key)
end
