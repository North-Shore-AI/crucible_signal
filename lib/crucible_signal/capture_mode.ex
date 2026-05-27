defmodule CrucibleSignal.CaptureMode do
  @moduledoc """
  Capture posture for a signal value.
  """

  @modes [:summary, :sample, :compressed_vector, :raw, :external_ref]

  @type t :: :summary | :sample | :compressed_vector | :raw | :external_ref

  def all, do: @modes

  def valid?(mode), do: mode in @modes

  def normalize(mode) when is_atom(mode) do
    if valid?(mode), do: {:ok, mode}, else: {:error, {:unknown_capture_mode, mode}}
  end

  def normalize(mode) when is_binary(mode) do
    mode
    |> String.replace("-", "_")
    |> String.to_existing_atom()
    |> normalize()
  rescue
    ArgumentError -> {:error, {:unknown_capture_mode, mode}}
  end

  def normalize(mode), do: {:error, {:unknown_capture_mode, mode}}
end
