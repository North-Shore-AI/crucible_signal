defmodule Crucible.ArtifactRef do
  @moduledoc """
  V4 pointer to a non-tensor trace artifact.
  """

  @derive Jason.Encoder
  defstruct [:uri, :digest, :byte_size, :media_type, metadata: %{}]

  @type t :: %__MODULE__{}
end
