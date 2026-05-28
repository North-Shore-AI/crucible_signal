defmodule Crucible.TensorSummary do
  @moduledoc """
  V4/V5 canonical bounded tensor summary.
  """

  alias CrucibleSignal.DType

  @derive Jason.Encoder
  defstruct shape: [],
            rank: 0,
            dtype: :unknown,
            min: nil,
            max: nil,
            mean: nil,
            stddev: nil,
            norm_l2: nil,
            nan_count: 0,
            positive_infinity_count: 0,
            negative_infinity_count: 0,
            entropy: nil,
            top_k: nil,
            digest: nil

  @type t :: %__MODULE__{}

  def compute(value, opts \\ [])

  def compute(%Nx.Tensor{} = tensor, opts) do
    values =
      tensor
      |> Nx.reshape({:auto})
      |> Nx.to_flat_list()

    build(values, Tuple.to_list(Nx.shape(tensor)), DType.from_nx_type(Nx.type(tensor)), opts)
  end

  def compute(values, opts) when is_list(values) do
    flat = List.flatten(values)
    build(flat, infer_shape(values), infer_dtype(flat), opts)
  end

  def compute(value, opts) when is_number(value),
    do: build([value], [], infer_dtype([value]), opts)

  defp build(values, shape, dtype, opts) do
    classes = Enum.map(values, &classify/1)
    finite = finite_values(values)
    probabilities = probabilities(finite)

    %__MODULE__{
      shape: shape,
      rank: length(shape),
      dtype: dtype,
      min: min_or_nil(finite),
      max: max_or_nil(finite),
      mean: mean(finite),
      stddev: stddev(finite),
      norm_l2: norm_l2(finite),
      nan_count: Enum.count(classes, &(&1 == :nan)),
      positive_infinity_count: Enum.count(classes, &(&1 == :positive_infinity)),
      negative_infinity_count: Enum.count(classes, &(&1 == :negative_infinity)),
      entropy: if(Keyword.get(opts, :entropy, false), do: entropy(probabilities)),
      top_k: top_k(finite, probabilities, Keyword.get(opts, :top_k, 10)),
      digest: digest(shape, dtype, values)
    }
  end

  defp finite_values(values) do
    values
    |> Enum.filter(&is_number/1)
    |> Enum.reject(&nan_or_inf?/1)
    |> Enum.map(&(&1 * 1.0))
  end

  defp classify(value) when is_integer(value), do: :finite

  defp classify(value) when is_float(value) do
    text = value |> Float.to_string() |> String.downcase()

    cond do
      text in ["nan"] -> :nan
      text in ["inf", "+inf", "infinity", "+infinity"] -> :positive_infinity
      text in ["-inf", "-infinity"] -> :negative_infinity
      true -> :finite
    end
  end

  defp classify(_value), do: :ignored

  defp nan_or_inf?(value) when is_integer(value), do: false

  defp nan_or_inf?(value) when is_float(value) do
    text = value |> Float.to_string() |> String.downcase()
    text in ["nan", "inf", "+inf", "-inf", "infinity", "-infinity"]
  end

  defp probabilities([]), do: []

  defp probabilities(values) do
    max_value = Enum.max(values)
    exps = Enum.map(values, &:math.exp(&1 - max_value))
    total = Enum.sum(exps)
    Enum.map(exps, &(&1 / total))
  end

  defp top_k(_values, _probabilities, count) when count in [nil, 0], do: []

  defp top_k(values, probabilities, count) do
    values
    |> Enum.with_index()
    |> Enum.map(fn {logit, index} ->
      %{token_id: index, logit: logit, probability: Enum.at(probabilities, index)}
    end)
    |> Enum.sort_by(& &1.logit, :desc)
    |> Enum.take(count)
  end

  defp entropy(nil), do: nil
  defp entropy([]), do: nil

  defp entropy(probabilities) do
    probabilities
    |> Enum.reject(&(&1 <= 0.0))
    |> Enum.map(&(-&1 * :math.log(&1)))
    |> Enum.sum()
  end

  defp min_or_nil([]), do: nil
  defp min_or_nil(values), do: Enum.min(values)

  defp max_or_nil([]), do: nil
  defp max_or_nil(values), do: Enum.max(values)

  defp mean([]), do: nil
  defp mean(values), do: Enum.sum(values) / length(values)

  defp stddev([]), do: nil
  defp stddev([_value]), do: 0.0

  defp stddev(values) do
    avg = mean(values)

    values
    |> Enum.map(&:math.pow(&1 - avg, 2))
    |> Enum.sum()
    |> Kernel./(length(values))
    |> :math.sqrt()
  end

  defp norm_l2([]), do: nil

  defp norm_l2(values) do
    values
    |> Enum.map(&(&1 * &1))
    |> Enum.sum()
    |> :math.sqrt()
  end

  defp digest(shape, dtype, values) do
    "sha256:" <>
      Base.encode16(:crypto.hash(:sha256, :erlang.term_to_binary({shape, dtype, values})),
        case: :lower
      )
  end

  defp infer_shape([]), do: [0]

  defp infer_shape([first | _rest] = values) when is_list(first) do
    [length(values) | infer_shape(first)]
  end

  defp infer_shape(values) when is_list(values), do: [length(values)]

  defp infer_dtype([]), do: :unknown

  defp infer_dtype(values) when is_list(values) do
    if Enum.all?(values, &is_integer/1), do: :s64, else: :f64
  end
end
