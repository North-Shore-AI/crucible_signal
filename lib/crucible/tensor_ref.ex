defmodule Crucible.TensorRef do
  @moduledoc """
  V4/V5 pointer to externally stored raw tensor bytes.
  """

  @derive Jason.Encoder
  defstruct [:uri, :digest, :shape, :dtype, :byte_size, :format]

  @type t :: %__MODULE__{}
end
