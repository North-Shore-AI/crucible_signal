defmodule CrucibleSignal.CapabilityStatus do
  @moduledoc """
  V5 capture/capability status vocabulary for signal attempts.
  """

  @statuses [
    :captured,
    :degraded,
    :unsupported,
    :unsupported_by_surface,
    :unsupported_by_backend,
    :unsupported_by_model_family,
    :blocked,
    :blocked_by_generation_pipeline,
    :failed,
    :failed_with_exception,
    :skipped
  ]

  @external_statuses Map.new(@statuses, fn status ->
                       {status |> Atom.to_string() |> String.replace("_", "-"), status}
                     end)

  @legacy_external_statuses %{
    "blocked-by-bumblebee-api" => :blocked,
    "blocked-by-axon-graph" => :blocked
  }

  @type t :: unquote(Enum.reduce(@statuses, &{:|, [], [&1, &2]}))

  @spec all() :: [t()]
  def all, do: @statuses

  @spec valid?(atom()) :: boolean()
  def valid?(status), do: status in @statuses

  @spec normalize(atom() | String.t()) ::
          {:ok, t()} | {:error, {:unknown_capability_status, term()}}
  def normalize(status) when is_atom(status) do
    status
    |> Atom.to_string()
    |> String.replace("_", "-")
    |> normalize_external(status)
  end

  def normalize(status) when is_binary(status) do
    status
    |> String.replace("_", "-")
    |> normalize_external(status)
  end

  def normalize(status), do: {:error, {:unknown_capability_status, status}}

  @spec class(t()) :: :captured | :degraded | :unsupported | :blocked | :failed | :skipped
  def class(:captured), do: :captured
  def class(:degraded), do: :degraded
  def class(:unsupported), do: :unsupported
  def class(:blocked), do: :blocked
  def class(:failed), do: :failed
  def class(:skipped), do: :skipped

  def class(status)
      when status in [
             :unsupported_by_surface,
             :unsupported_by_backend,
             :unsupported_by_model_family
           ],
      do: :unsupported

  def class(status)
      when status in [
             :blocked_by_generation_pipeline
           ],
      do: :blocked

  def class(:failed_with_exception), do: :failed

  defp normalize_external(key, original) do
    case Map.fetch(Map.merge(@legacy_external_statuses, @external_statuses), key) do
      {:ok, status} -> {:ok, status}
      :error -> {:error, {:unknown_capability_status, original}}
    end
  end
end
