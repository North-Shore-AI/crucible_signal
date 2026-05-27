defmodule CrucibleSignal.DType do
  @moduledoc """
  Canonical dtype names for signal metadata.
  """

  @types [
    :bf16,
    :f16,
    :f32,
    :f64,
    :s8,
    :s16,
    :s32,
    :s64,
    :u8,
    :u16,
    :u32,
    :u64,
    :bool,
    :string,
    :token,
    :unknown
  ]

  @type t :: atom()

  def all, do: @types

  def valid?(dtype), do: dtype in @types

  def normalize(nil), do: {:ok, nil}

  def normalize(dtype) when is_atom(dtype) do
    if valid?(dtype), do: {:ok, dtype}, else: {:error, {:unknown_dtype, dtype}}
  end

  def normalize(dtype) when is_binary(dtype) do
    dtype
    |> String.downcase()
    |> String.to_existing_atom()
    |> normalize()
  rescue
    ArgumentError -> {:error, {:unknown_dtype, dtype}}
  end

  def normalize(dtype), do: {:error, {:unknown_dtype, dtype}}

  def from_nx_type({:bf, 16}), do: :bf16
  def from_nx_type({:f, 16}), do: :f16
  def from_nx_type({:f, 32}), do: :f32
  def from_nx_type({:f, 64}), do: :f64
  def from_nx_type({:s, 8}), do: :s8
  def from_nx_type({:s, 16}), do: :s16
  def from_nx_type({:s, 32}), do: :s32
  def from_nx_type({:s, 64}), do: :s64
  def from_nx_type({:u, 8}), do: :u8
  def from_nx_type({:u, 16}), do: :u16
  def from_nx_type({:u, 32}), do: :u32
  def from_nx_type({:u, 64}), do: :u64
  def from_nx_type({:pred, _}), do: :bool
  def from_nx_type(_), do: :unknown
end
