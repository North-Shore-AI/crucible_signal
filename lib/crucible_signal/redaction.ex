defmodule CrucibleSignal.Redaction do
  @moduledoc """
  Redaction posture for signal records and tensor payloads.
  """

  @modes [:none, :summary_only, :redacted, :external_ref]

  @type t :: :none | :summary_only | :redacted | :external_ref

  def all, do: @modes

  def valid?(mode), do: mode in @modes

  def normalize(mode) when is_atom(mode) do
    if valid?(mode), do: {:ok, mode}, else: {:error, {:unknown_redaction, mode}}
  end

  def normalize(mode) when is_binary(mode) do
    mode
    |> String.replace("-", "_")
    |> String.to_existing_atom()
    |> normalize()
  rescue
    ArgumentError -> {:error, {:unknown_redaction, mode}}
  end

  def normalize(mode), do: {:error, {:unknown_redaction, mode}}
end
