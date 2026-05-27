alias CrucibleSignal.SignalRef

refs = [
  SignalRef.for_final_logits(trace_id: "trace-example"),
  SignalRef.for_layer_residual(4, trace_id: "trace-example", capture_mode: :compressed_vector),
  SignalRef.for_attention_map(4, 2, trace_id: "trace-example"),
  SignalRef.for_mlp_gate(4, trace_id: "trace-example"),
  SignalRef.for_norm_telemetry(4, trace_id: "trace-example"),
  SignalRef.for_moe_router(4, trace_id: "trace-example"),
  SignalRef.for_world_model(trace_id: "trace-example"),
  SignalRef.for_verifier(trace_id: "trace-example"),
  SignalRef.for_logit_lens(8, trace_id: "trace-example"),
  SignalRef.for_kv_cache(0, trace_id: "trace-example"),
  SignalRef.for_decoded_text(trace_id: "trace-example")
]

Enum.each(refs, &SignalRef.validate!/1)

IO.puts(Jason.encode!(%{
  ok: true,
  example: "signal_factory_mock",
  count: length(refs),
  signal_types: Enum.map(refs, & &1.signal_type)
}))
