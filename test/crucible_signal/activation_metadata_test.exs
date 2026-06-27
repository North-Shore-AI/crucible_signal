defmodule CrucibleSignal.ActivationMetadataTest do
  use ExUnit.Case, async: true

  alias CrucibleSignal.{ActivationMetadata, SignalRef, SignalSpec}

  @canonical_names [
    "hook_tokens",
    "hook_embed",
    "hook_pos_embed",
    "blocks.0.hook_resid_pre",
    "blocks.0.attn.hook_q",
    "blocks.0.attn.hook_k",
    "blocks.0.attn.hook_v",
    "blocks.0.attn.hook_attn_scores",
    "blocks.0.attn.hook_pattern",
    "blocks.0.attn.hook_z",
    "blocks.0.attn.hook_result",
    "blocks.0.hook_attn_out",
    "blocks.0.hook_resid_mid",
    "blocks.0.mlp.hook_pre",
    "blocks.0.mlp.hook_post",
    "blocks.0.hook_mlp_out",
    "blocks.0.hook_resid_post",
    "ln_final.hook_scale",
    "ln_final.hook_normalized",
    "unembed.hook_logits"
  ]

  test "parses every canonical TransformerLens activation name" do
    for name <- @canonical_names do
      axes = ActivationMetadata.default_axes(name)

      assert {:ok, parsed} = ActivationMetadata.parse_name(name)
      assert parsed.name == name

      assert {:ok, metadata} =
               ActivationMetadata.normalize(%{
                 activation_name: name,
                 axes: axes,
                 capture_mode: "summary"
               })

      assert metadata.activation_name == name
      assert metadata.axes == axes
      assert metadata.capture_mode == :summary
    end
  end

  test "rejects activation claims without axes" do
    for name <- [
          "blocks.0.attn.hook_q",
          "blocks.0.attn.hook_k",
          "blocks.0.attn.hook_v",
          "blocks.0.attn.hook_pattern",
          "blocks.0.attn.hook_result"
        ] do
      assert {:error, {:missing_activation_axes, ^name}} =
               ActivationMetadata.normalize(%{activation_name: name})
    end
  end

  test "rejects semantically invalid attention axes" do
    assert {:error, {:invalid_activation_axes, "blocks.0.attn.hook_q", :missing_head_axis}} =
             ActivationMetadata.normalize(%{
               activation_name: "blocks.0.attn.hook_q",
               axes: [:batch, :pos, :d_head]
             })

    assert {:error,
            {:invalid_activation_axes, "blocks.0.attn.hook_pattern",
             :missing_attention_position_axes}} =
             ActivationMetadata.normalize(%{
               activation_name: "blocks.0.attn.hook_pattern",
               axes: [:batch, :head, :pos]
             })

    assert {:error,
            {:invalid_activation_axes, "blocks.0.attn.hook_result", :missing_head_result_axes}} =
             ActivationMetadata.normalize(%{
               activation_name: "blocks.0.attn.hook_result",
               axes: [:batch, :pos, :head, :d_head]
             })
  end

  test "normalizes string-key metadata without creating arbitrary atom keys" do
    external_key = "external_activation_key_#{System.unique_integer([:positive])}"

    refute existing_atom?(external_key)

    assert {:ok, metadata} =
             ActivationMetadata.normalize(%{
               "activation_name" => "blocks.2.attn.hook_k",
               "axes" => ["batch", "pos", "kv_head", "d_head"],
               "capture_mode" => "raw",
               external_key => "kept"
             })

    assert metadata.activation_name == "blocks.2.attn.hook_k"
    assert metadata.component == :attn
    assert metadata.layer_index == 2
    assert metadata.axes == [:batch, :pos, :kv_head, :d_head]
    assert metadata.capture_mode == :raw
    assert metadata[external_key] == "kept"
    refute existing_atom?(external_key)
  end

  test "signal constructors validate activation metadata" do
    assert_raise ArgumentError, ~r/missing_activation_axes/, fn ->
      Crucible.SignalRecord.new!(
        signal_id: "q",
        trace_id: "trace",
        signal_type: :attention_q,
        metadata: %{activation_name: "blocks.0.attn.hook_q"}
      )
    end

    spec =
      SignalSpec.new!(
        signal_type: :attention_q,
        metadata: %{
          activation_name: "blocks.0.attn.hook_q",
          axes: [:batch, :pos, :head, :d_head]
        }
      )

    assert spec.metadata.component == :attn
  end

  test "builds activation signal refs with default axes and signal type" do
    ref =
      SignalRef.for_activation("blocks.3.attn.hook_pattern",
        trace_id: "trace-activation",
        signal_id: "attn-pattern"
      )

    assert ref.signal_type == :attention_weights
    assert ref.layer_index == nil
    assert ref.metadata.activation_name == "blocks.3.attn.hook_pattern"
    assert ref.metadata.layer_index == 3
    assert ref.metadata.axes == [:batch, :head, :dest_pos, :src_pos]
  end

  defp existing_atom?(value) do
    _ = String.to_existing_atom(value)
    true
  rescue
    ArgumentError -> false
  end
end
