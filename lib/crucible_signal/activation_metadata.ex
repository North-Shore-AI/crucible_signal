defmodule CrucibleSignal.ActivationMetadata do
  @moduledoc """
  Canonical metadata helpers for TransformerLens-style activation captures.

  `crucible_signal` intentionally keeps this as a lightweight schema helper
  instead of depending on provider or analysis libraries. Rich analysis lives in
  higher-level packages, while signal/tap/trace contracts can still validate the
  stable activation namespace, axes, and capture posture.
  """

  alias CrucibleSignal.{CaptureMode, SafeTerms, SignalType}

  @type axis ::
          :batch
          | :pos
          | :layer
          | :head
          | :kv_head
          | :dest_pos
          | :src_pos
          | :d_model
          | :d_head
          | :d_mlp
          | :d_vocab
          | :scale
          | :component
          | :direction
          | :rank
          | :expert
          | :decode_step

  @type component :: :input | :block | :attn | :mlp | :final_norm | :unembed

  @axes [
    :batch,
    :pos,
    :layer,
    :head,
    :kv_head,
    :dest_pos,
    :src_pos,
    :d_model,
    :d_head,
    :d_mlp,
    :d_vocab,
    :scale,
    :component,
    :direction,
    :rank,
    :expert,
    :decode_step
  ]

  @axis_by_string Map.new(@axes, &{Atom.to_string(&1), &1})
  @components [:input, :block, :attn, :mlp, :final_norm, :unembed]
  @component_by_string Map.new(@components, &{Atom.to_string(&1), &1})

  @doc "Returns the canonical activation-name templates owned by the signal contract."
  @spec templates() :: [String.t()]
  def templates do
    [
      "hook_tokens",
      "hook_embed",
      "hook_pos_embed",
      "blocks.{layer}.hook_resid_pre",
      "blocks.{layer}.attn.hook_q",
      "blocks.{layer}.attn.hook_k",
      "blocks.{layer}.attn.hook_v",
      "blocks.{layer}.attn.hook_attn_scores",
      "blocks.{layer}.attn.hook_pattern",
      "blocks.{layer}.attn.hook_z",
      "blocks.{layer}.attn.hook_result",
      "blocks.{layer}.hook_attn_out",
      "blocks.{layer}.hook_resid_mid",
      "blocks.{layer}.mlp.hook_pre",
      "blocks.{layer}.mlp.hook_post",
      "blocks.{layer}.hook_mlp_out",
      "blocks.{layer}.hook_resid_post",
      "ln_final.hook_scale",
      "ln_final.hook_normalized",
      "unembed.hook_logits"
    ]
  end

  @doc "Normalizes and validates activation metadata while preserving unrelated keys."
  @spec normalize(map() | keyword() | nil) :: {:ok, map()} | {:error, term()}
  def normalize(nil), do: {:ok, %{}}
  def normalize(metadata) when is_list(metadata), do: metadata |> Map.new() |> normalize()
  def normalize(%_struct{} = metadata), do: metadata |> Map.from_struct() |> normalize()

  def normalize(metadata) when is_map(metadata) do
    metadata = SafeTerms.normalize_attrs(metadata)

    with {:ok, parsed} <- normalize_activation_name(Map.get(metadata, :activation_name)),
         {:ok, component} <- normalize_component(Map.get(metadata, :component), parsed),
         {:ok, axes} <- normalize_axes(Map.get(metadata, :axes)),
         {:ok, capture_mode} <- normalize_capture_mode(Map.get(metadata, :capture_mode)),
         {:ok, layer_index} <- normalize_layer_index(Map.get(metadata, :layer_index), parsed),
         {:ok, head_index} <-
           normalize_non_negative_integer(Map.get(metadata, :head_index), :head_index),
         {:ok, kv_head_index} <-
           normalize_non_negative_integer(Map.get(metadata, :kv_head_index), :kv_head_index),
         {:ok, decode_step} <-
           normalize_non_negative_integer(Map.get(metadata, :decode_step), :decode_step),
         {:ok, token_index} <- normalize_token_index(Map.get(metadata, :token_index)),
         :ok <- validate_axes(parsed, axes),
         :ok <- validate_signal_type(Map.get(metadata, :signal_type)) do
      metadata =
        metadata
        |> put_present(:activation_name, parsed && parsed.name)
        |> put_present(:component, component)
        |> put_present(:axes, axes)
        |> put_present(:capture_mode, capture_mode)
        |> put_present(:layer_index, layer_index)
        |> put_present(:head_index, head_index)
        |> put_present(:kv_head_index, kv_head_index)
        |> put_present(:decode_step, decode_step)
        |> put_present(:token_index, token_index)

      {:ok, metadata}
    end
  end

  def normalize(metadata), do: {:error, {:invalid_activation_metadata, metadata}}

  @doc "Normalizes activation metadata or raises `ArgumentError`."
  @spec normalize!(map() | keyword() | nil) :: map()
  def normalize!(metadata) do
    case normalize(metadata) do
      {:ok, normalized} ->
        normalized

      {:error, reason} ->
        raise ArgumentError, "invalid activation metadata: #{inspect(reason)}"
    end
  end

  @doc "Parses a canonical activation name without provider-specific assumptions."
  @spec parse_name(String.t()) :: {:ok, map()} | {:error, term()}
  def parse_name(name) when is_binary(name) do
    cond do
      name in ["hook_tokens", "hook_embed", "hook_pos_embed"] ->
        {:ok, %{name: name, component: :input, layer_index: nil, hook: name}}

      name in ["ln_final.hook_scale", "ln_final.hook_normalized"] ->
        {:ok,
         %{
           name: name,
           component: :final_norm,
           layer_index: :final,
           hook: String.split(name, ".") |> List.last()
         }}

      name == "unembed.hook_logits" ->
        {:ok, %{name: name, component: :unembed, layer_index: :final, hook: "hook_logits"}}

      match = Regex.run(~r/^blocks\.(\d+)\.attn\.(hook_[a-z_]+)$/, name) ->
        [_, layer, hook] = match
        parse_block_hook(name, String.to_integer(layer), :attn, hook)

      match = Regex.run(~r/^blocks\.(\d+)\.mlp\.(hook_[a-z_]+)$/, name) ->
        [_, layer, hook] = match
        parse_block_hook(name, String.to_integer(layer), :mlp, hook)

      match = Regex.run(~r/^blocks\.(\d+)\.(hook_[a-z_]+)$/, name) ->
        [_, layer, hook] = match
        parse_block_hook(name, String.to_integer(layer), :block, hook)

      true ->
        {:error, :unknown_activation_name}
    end
  end

  def parse_name(name), do: {:error, {:invalid_activation_name, name}}

  @doc "Returns default axes for a canonical activation name."
  @spec default_axes(String.t()) :: [axis()]
  def default_axes("hook_tokens"), do: [:batch, :pos]
  def default_axes("hook_embed"), do: [:batch, :pos, :d_model]
  def default_axes("hook_pos_embed"), do: [:batch, :pos, :d_model]
  def default_axes("ln_final.hook_scale"), do: [:batch, :pos, :scale]
  def default_axes("ln_final.hook_normalized"), do: [:batch, :pos, :d_model]
  def default_axes("unembed.hook_logits"), do: [:batch, :pos, :d_vocab]

  def default_axes(name) when is_binary(name) do
    cond do
      String.ends_with?(name, ".attn.hook_attn_scores") -> [:batch, :head, :dest_pos, :src_pos]
      String.ends_with?(name, ".attn.hook_pattern") -> [:batch, :head, :dest_pos, :src_pos]
      String.ends_with?(name, ".attn.hook_q") -> [:batch, :pos, :head, :d_head]
      String.ends_with?(name, ".attn.hook_k") -> [:batch, :pos, :kv_head, :d_head]
      String.ends_with?(name, ".attn.hook_v") -> [:batch, :pos, :kv_head, :d_head]
      String.ends_with?(name, ".attn.hook_z") -> [:batch, :pos, :head, :d_head]
      String.ends_with?(name, ".attn.hook_result") -> [:batch, :pos, :head, :d_model]
      String.ends_with?(name, ".mlp.hook_pre") -> [:batch, :pos, :d_mlp]
      String.ends_with?(name, ".mlp.hook_post") -> [:batch, :pos, :d_mlp]
      String.contains?(name, ".hook_") -> [:batch, :pos, :d_model]
      true -> []
    end
  end

  def default_axes(_name), do: []

  @doc "Returns the default signal type for a canonical activation name."
  @spec default_signal_type(String.t()) :: SignalType.t()
  def default_signal_type(name) do
    cond do
      name == "hook_tokens" -> :input_ids
      name in ["hook_embed", "hook_pos_embed"] -> :token_embeddings
      name == "ln_final.hook_scale" -> :norm_telemetry
      name == "ln_final.hook_normalized" -> :norm_telemetry
      name == "unembed.hook_logits" -> :final_logits
      String.ends_with?(name, ".attn.hook_q") -> :attention_q
      String.ends_with?(name, ".attn.hook_k") -> :attention_k
      String.ends_with?(name, ".attn.hook_v") -> :attention_v
      String.ends_with?(name, ".attn.hook_attn_scores") -> :attention_scores
      String.ends_with?(name, ".attn.hook_pattern") -> :attention_weights
      String.ends_with?(name, ".attn.hook_z") -> :head_outputs
      String.ends_with?(name, ".attn.hook_result") -> :head_outputs
      String.ends_with?(name, ".mlp.hook_pre") -> :mlp_activation
      String.ends_with?(name, ".mlp.hook_post") -> :mlp_activation
      String.contains?(name, "hook_resid") -> :residual_stream
      String.ends_with?(name, ".hook_attn_out") -> :residual_stream
      String.ends_with?(name, ".hook_mlp_out") -> :residual_stream
      true -> :hidden_state
    end
  end

  @doc "Adds canonical activation fields to an existing metadata map."
  @spec put_activation(map() | keyword() | nil, String.t(), keyword() | map()) :: map()
  def put_activation(metadata, activation_name, attrs \\ []) do
    {:ok, parsed} = parse_name(activation_name)

    attrs = SafeTerms.normalize_attrs(attrs)
    metadata = normalize!(metadata)

    metadata
    |> Map.put(:activation_name, parsed.name)
    |> Map.put_new(:component, parsed.component)
    |> Map.put_new(:layer_index, parsed.layer_index)
    |> Map.put_new(:axes, default_axes(parsed.name))
    |> Map.merge(attrs)
    |> normalize!()
  end

  defp normalize_activation_name(nil), do: {:ok, nil}

  defp normalize_activation_name(name) when is_binary(name) do
    case parse_name(name) do
      {:ok, parsed} -> {:ok, parsed}
      {:error, reason} -> {:error, {:invalid_activation_name, name, reason}}
    end
  end

  defp normalize_activation_name(name), do: {:error, {:invalid_activation_name, name}}

  defp normalize_component(nil, nil), do: {:ok, nil}
  defp normalize_component(nil, parsed), do: {:ok, parsed.component}
  defp normalize_component(component, _parsed) when component in @components, do: {:ok, component}

  defp normalize_component(component, _parsed) when is_binary(component) do
    component =
      component
      |> String.trim()
      |> String.replace("-", "_")

    case Map.fetch(@component_by_string, component) do
      {:ok, component} -> {:ok, component}
      :error -> {:error, {:invalid_activation_component, component}}
    end
  end

  defp normalize_component(component, _parsed),
    do: {:error, {:invalid_activation_component, component}}

  defp normalize_axes(nil), do: {:ok, nil}

  defp normalize_axes(axes) when is_list(axes) do
    axes
    |> Enum.reduce_while({:ok, []}, fn axis, {:ok, acc} ->
      case normalize_axis(axis) do
        {:ok, axis} -> {:cont, {:ok, [axis | acc]}}
        {:error, reason} -> {:halt, {:error, reason}}
      end
    end)
    |> case do
      {:ok, axes} -> {:ok, Enum.reverse(axes)}
      {:error, reason} -> {:error, reason}
    end
  end

  defp normalize_axes(axis), do: {:error, {:invalid_activation_axes, axis}}

  defp normalize_axis(axis) when axis in @axes, do: {:ok, axis}

  defp normalize_axis(axis) when is_binary(axis) do
    axis =
      axis
      |> String.trim()
      |> String.replace("-", "_")

    case Map.fetch(@axis_by_string, axis) do
      {:ok, axis} -> {:ok, axis}
      :error -> {:error, {:invalid_activation_axis, axis}}
    end
  end

  defp normalize_axis(axis), do: {:error, {:invalid_activation_axis, axis}}

  defp normalize_capture_mode(nil), do: {:ok, nil}

  defp normalize_capture_mode(capture_mode) do
    case CaptureMode.normalize(capture_mode) do
      {:ok, capture_mode} -> {:ok, capture_mode}
      {:error, reason} -> {:error, reason}
    end
  end

  defp normalize_layer_index(nil, nil), do: {:ok, nil}
  defp normalize_layer_index(nil, parsed), do: {:ok, parsed.layer_index}
  defp normalize_layer_index(:final, _parsed), do: {:ok, :final}
  defp normalize_layer_index("final", _parsed), do: {:ok, :final}

  defp normalize_layer_index(index, _parsed) when is_integer(index) and index >= 0,
    do: {:ok, index}

  defp normalize_layer_index(index, _parsed),
    do: {:error, {:invalid_activation_layer_index, index}}

  defp normalize_non_negative_integer(nil, _field), do: {:ok, nil}

  defp normalize_non_negative_integer(value, _field) when is_integer(value) and value >= 0,
    do: {:ok, value}

  defp normalize_non_negative_integer(value, field), do: {:error, {:"invalid_#{field}", value}}

  defp normalize_token_index(nil), do: {:ok, nil}
  defp normalize_token_index(value) when is_integer(value), do: {:ok, value}
  defp normalize_token_index(value), do: {:error, {:invalid_token_index, value}}

  defp validate_axes(nil, _axes), do: :ok
  defp validate_axes(%{name: name}, nil), do: {:error, {:missing_activation_axes, name}}
  defp validate_axes(%{name: name}, []), do: {:error, {:missing_activation_axes, name}}

  defp validate_axes(%{name: name}, axes) do
    cond do
      String.ends_with?(name, ".attn.hook_q") and :head not in axes ->
        {:error, {:invalid_activation_axes, name, :missing_head_axis}}

      String.ends_with?(name, ".attn.hook_k") and
          not Enum.any?(axes, &(&1 in [:head, :kv_head])) ->
        {:error, {:invalid_activation_axes, name, :missing_key_head_axis}}

      String.ends_with?(name, ".attn.hook_v") and
          not Enum.any?(axes, &(&1 in [:head, :kv_head])) ->
        {:error, {:invalid_activation_axes, name, :missing_value_head_axis}}

      String.ends_with?(name, ".attn.hook_attn_scores") and
          not Enum.all?([:dest_pos, :src_pos], &(&1 in axes)) ->
        {:error, {:invalid_activation_axes, name, :missing_attention_position_axes}}

      String.ends_with?(name, ".attn.hook_pattern") and
          not Enum.all?([:dest_pos, :src_pos], &(&1 in axes)) ->
        {:error, {:invalid_activation_axes, name, :missing_attention_position_axes}}

      String.ends_with?(name, ".attn.hook_result") and
          not Enum.all?([:head, :d_model], &(&1 in axes)) ->
        {:error, {:invalid_activation_axes, name, :missing_head_result_axes}}

      true ->
        :ok
    end
  end

  defp validate_signal_type(nil), do: :ok

  defp validate_signal_type(signal_type) do
    case SignalType.normalize(signal_type) do
      {:ok, _signal_type} -> :ok
      {:error, reason} -> {:error, reason}
    end
  end

  defp parse_block_hook(name, layer, :attn, hook)
       when hook in [
              "hook_q",
              "hook_k",
              "hook_v",
              "hook_attn_scores",
              "hook_pattern",
              "hook_z",
              "hook_result"
            ] do
    {:ok, %{name: name, component: :attn, layer_index: layer, hook: hook}}
  end

  defp parse_block_hook(name, layer, :mlp, hook) when hook in ["hook_pre", "hook_post"] do
    {:ok, %{name: name, component: :mlp, layer_index: layer, hook: hook}}
  end

  defp parse_block_hook(name, layer, :block, hook)
       when hook in [
              "hook_resid_pre",
              "hook_resid_mid",
              "hook_resid_post",
              "hook_attn_out",
              "hook_mlp_out"
            ] do
    {:ok, %{name: name, component: :block, layer_index: layer, hook: hook}}
  end

  defp parse_block_hook(_name, _layer, _component, hook), do: {:error, {:unknown_hook, hook}}

  defp put_present(metadata, _key, nil), do: metadata
  defp put_present(metadata, key, value), do: Map.put(metadata, key, value)
end
