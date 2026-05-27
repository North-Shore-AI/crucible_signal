alias CrucibleSignal.TensorSummary

summary =
  Nx.iota({32}, type: :f32)
  |> TensorSummary.from_nx(top_k: 3, entropy: true)

IO.puts(Jason.encode!(%{
  ok: true,
  example: "tensor_summary_live",
  backend: inspect(Nx.default_backend()),
  count: summary.count,
  top_k: summary.top_k
}))
