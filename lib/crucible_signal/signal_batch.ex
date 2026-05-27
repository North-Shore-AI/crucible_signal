defmodule CrucibleSignal.SignalBatch do
  @moduledoc """
  Ordered collection of signal references.
  """

  alias CrucibleSignal.SignalRef

  @derive Jason.Encoder
  defstruct batch_id: nil, refs: [], metadata: %{}

  @type t :: %__MODULE__{}

  def new(refs, attrs \\ []) when is_list(refs) do
    if Enum.all?(refs, &match?(%SignalRef{}, &1)) do
      {:ok,
       %__MODULE__{
         batch_id: Keyword.get(attrs, :batch_id, "batch:#{System.unique_integer([:positive])}"),
         refs: refs,
         metadata: Keyword.get(attrs, :metadata, %{})
       }}
    else
      {:error, :invalid_signal_ref}
    end
  end

  def new!(refs, attrs \\ []) do
    case new(refs, attrs) do
      {:ok, batch} -> batch
      {:error, reason} -> raise ArgumentError, "invalid signal batch: #{inspect(reason)}"
    end
  end
end
