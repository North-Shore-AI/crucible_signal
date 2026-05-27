defmodule CrucibleSignal.TensorShape do
  @moduledoc """
  Serializable tensor shape metadata.
  """

  @derive Jason.Encoder
  defstruct dims: [], rank: 0, element_count: 0

  @type dim :: non_neg_integer() | nil
  @type t :: %__MODULE__{
          dims: [dim()],
          rank: non_neg_integer(),
          element_count: non_neg_integer() | nil
        }

  @doc "Builds a shape struct from a tuple or list of dimensions."
  def new!(dims) when is_tuple(dims), do: dims |> Tuple.to_list() |> new!()

  def new!(dims) when is_list(dims) do
    unless Enum.all?(dims, &valid_dim?/1) do
      raise ArgumentError, "invalid tensor dimensions: #{inspect(dims)}"
    end

    %__MODULE__{
      dims: dims,
      rank: length(dims),
      element_count: element_count(dims)
    }
  end

  def new!(other), do: raise(ArgumentError, "expected tensor dimensions, got: #{inspect(other)}")

  defp valid_dim?(nil), do: true
  defp valid_dim?(dim), do: is_integer(dim) and dim >= 0

  defp element_count(dims) do
    if Enum.any?(dims, &is_nil/1), do: nil, else: Enum.product(dims)
  end
end
