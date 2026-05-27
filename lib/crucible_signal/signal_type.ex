defmodule CrucibleSignal.SignalType do
  @moduledoc """
  Canonical forward-pass signal classes.
  """

  @types [
    :embeddings,
    :early_residuals,
    :attention_q,
    :attention_k,
    :attention_v,
    :attention_maps,
    :head_outputs,
    :mlp_gates,
    :middle_residuals,
    :layer_trajectory,
    :norm_telemetry,
    :moe_router_logits,
    :kv_cache,
    :kv_cache_state,
    :world_model_state,
    :verifier_signal,
    :logit_lens_intermediate,
    :late_residuals,
    :intermediate_logits,
    :final_logits,
    :decoded_text
  ]

  @type t :: unquote(Enum.reduce(@types, &{:|, [], [&1, &2]}))

  @doc "Returns every canonical signal type."
  def all, do: @types

  @doc "Returns true when the atom is a known signal type."
  def valid?(type), do: type in @types

  @doc "Normalizes atom or string signal type input."
  def normalize(type) when is_atom(type) do
    type = alias_type(type)
    if valid?(type), do: {:ok, type}, else: {:error, {:unknown_signal_type, type}}
  end

  def normalize(type) when is_binary(type) do
    type
    |> String.replace("-", "_")
    |> String.to_existing_atom()
    |> normalize()
  rescue
    ArgumentError -> {:error, {:unknown_signal_type, type}}
  end

  def normalize(type), do: {:error, {:unknown_signal_type, type}}

  defp alias_type(:kv_cache), do: :kv_cache_state
  defp alias_type(:intermediate_logits), do: :logit_lens_intermediate
  defp alias_type(type), do: type
end
