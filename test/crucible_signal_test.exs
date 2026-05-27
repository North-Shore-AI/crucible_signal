defmodule CrucibleSignalTest do
  use ExUnit.Case
  doctest CrucibleSignal

  alias CrucibleSignal.{
    Capability,
    CaptureMode,
    Operation,
    SignalBatch,
    SignalRef,
    SignalSpec,
    SignalType,
    TensorShape,
    TensorSummary
  }

  test "exposes package version" do
    assert CrucibleSignal.version() == "0.1.0"
  end

  test "covers the full forward-pass signal matrix" do
    assert CrucibleSignal.signal_types() == [
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

    assert {:ok, :attention_q} = SignalType.normalize("attention-q")
  end

  test "covers the operation matrix" do
    assert CrucibleSignal.operations() == [
             :read,
             :probe,
             :route_on,
             :fuse,
             :gate,
             :uncertainty,
             :control_vector,
             :shared_memory,
             :verifier_signal,
             :steer_model
           ]

    assert {:ok, :route_on} = Operation.normalize("route-on")
    assert CaptureMode.all() == [:summary, :sample, :compressed_vector, :raw, :external_ref]
  end

  test "builds validated signal refs" do
    ref =
      SignalRef.new!(
        trace_id: "trace-1",
        signal_id: "sig-1",
        signal_type: :final_logits,
        model_ref: "qwen3:fixture",
        layer_index: :final,
        token_index: -1,
        dtype: :f32,
        shape: {1, 8},
        capture_mode: :summary,
        metadata: %{source: "test"}
      )

    assert ref.signal_type == :final_logits
    assert ref.shape == %TensorShape{dims: [1, 8], rank: 2, element_count: 8}

    assert {:error, {:unknown_signal_type, :missing}} =
             SignalRef.new(trace_id: "t", signal_id: "s", signal_type: :missing)
  end

  test "builds validated signal specs and batches" do
    spec =
      SignalSpec.new!(
        id: "tap-final-logits",
        signal_type: "final_logits",
        operations: [:read, "route-on"],
        capture_mode: :summary,
        layers: [:final],
        tokens: [-1]
      )

    assert spec.operations == [:read, :route_on]

    ref = SignalRef.new!(trace_id: "trace-1", signal_id: "sig-1", signal_type: :embeddings)
    batch = SignalBatch.new!([ref], batch_id: "batch-1")

    assert batch.refs == [ref]
  end

  test "summarizes plain numeric fixtures deterministically" do
    summary = TensorSummary.summarize([[1, 2], [3, 4]], top_k: 2, entropy: true)

    assert summary.shape == %TensorShape{dims: [2, 2], rank: 2, element_count: 4}
    assert summary.dtype == :s64
    assert summary.count == 4
    assert summary.finite_count == 4
    assert summary.min == 1.0
    assert summary.max == 4.0
    assert summary.mean == 2.5
    assert_in_delta summary.stddev, 1.1180, 0.001
    assert_in_delta summary.l2_norm, 5.4772, 0.001
    assert summary.top_k == [4.0, 3.0]
    assert is_binary(summary.checksum)
    assert is_float(summary.entropy)
  end

  test "summarizes Nx tensors" do
    summary =
      Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: :f32)
      |> TensorSummary.summarize(top_k: 1)

    assert summary.shape == %TensorShape{dims: [2, 2], rank: 2, element_count: 4}
    assert summary.dtype == :f32
    assert summary.top_k == [4.0]
  end

  test "encodes structs to JSON" do
    ref = SignalRef.new!(trace_id: "trace-1", signal_id: "sig-1", signal_type: :kv_cache)

    assert {:ok, json} = Jason.encode(ref)
    assert {:ok, decoded} = Jason.decode(json)
    assert decoded["trace_id"] == "trace-1"
    assert decoded["signal_type"] == "kv_cache_state"
  end

  test "builds factory refs for added signal classes" do
    base = [trace_id: "trace-2", model_ref: "model:fixture"]

    assert SignalRef.for_moe_router(3, base).signal_type == :moe_router_logits
    assert SignalRef.for_world_model(base).signal_type == :world_model_state
    assert SignalRef.for_verifier(base).signal_type == :verifier_signal
    assert SignalRef.for_logit_lens(8, base).signal_type == :logit_lens_intermediate
    assert SignalRef.for_kv_cache(4, base).signal_type == :kv_cache_state

    assert SignalRef.for_layer_residual(4, Keyword.put(base, :capture_mode, :compressed_vector)).capture_mode ==
             :compressed_vector
  end

  test "merges count-compatible summaries through partial summaries" do
    left = TensorSummary.from_list([1, 2], entropy: false)
    right = TensorSummary.from_list([3, 4], entropy: false)

    assert {:ok, merged} = TensorSummary.merge(left, right)
    assert merged.count == 4
    assert merged.finite_count == 4
    assert merged.min == 1.0
    assert merged.max == 4.0
    assert merged.mean == 2.5

    entropy_summary = TensorSummary.from_list([1, 2], entropy: true)
    assert {:error, {:unmergeable_metric, :entropy}} = TensorSummary.merge(entropy_summary, right)
  end

  test "operations describe data contracts without tap preconditions" do
    assert {:ok, %{required_inputs: [:signal_type, :summary_or_vector]}} =
             Operation.describe(:route_on)

    assert {:ok, [:steering_plan]} = Operation.required_inputs("steer-model")
  end

  test "describes adapter capabilities" do
    capability =
      Capability.new!(
        signal_type: :attention_maps,
        operations: [:read, :probe, :uncertainty],
        capture_modes: [:summary, :sample],
        adapter: :bumblebee,
        model_family: :qwen3
      )

    assert Capability.supports?(capability, :probe)
    assert Capability.supports_capture?(capability, "sample")
    refute Capability.supports?(capability, :fuse)
  end
end
