defmodule CrucibleSignal.TensorSummary do
  @moduledoc """
  Bounded deterministic summaries for tensors and numeric fixtures.
  """

  alias CrucibleSignal.{DType, TensorShape}

  @derive Jason.Encoder
  defstruct shape: nil,
            dtype: :unknown,
            count: 0,
            finite_count: 0,
            nan_count: 0,
            positive_infinity_count: 0,
            negative_infinity_count: 0,
            min: nil,
            max: nil,
            mean: nil,
            stddev: nil,
            l2_norm: nil,
            top_k: [],
            entropy: nil,
            checksum: nil

  @type t :: %__MODULE__{}

  @doc """
  Summarizes an Nx tensor or nested numeric list.

  Options:

    * `:top_k` - number of largest finite values to keep. Defaults to `5`.
    * `:entropy` - when true, computes softmax entropy over finite values.
    * `:checksum` - when true, includes a SHA-256 checksum. Defaults to `true`.

  """
  def summarize(value, opts \\ [])

  def summarize(%Nx.Tensor{} = tensor, opts) do
    values =
      tensor
      |> Nx.reshape({:auto})
      |> Nx.to_flat_list()

    build(values, TensorShape.new!(Nx.shape(tensor)), DType.from_nx_type(Nx.type(tensor)), opts)
  end

  def summarize(values, opts) when is_list(values) do
    build(List.flatten(values), TensorShape.new!(infer_shape(values)), infer_dtype(values), opts)
  end

  def summarize(value, opts) when is_number(value) do
    build([value], TensorShape.new!([]), infer_dtype([value]), opts)
  end

  defp build(values, shape, dtype, opts) do
    top_k_count = Keyword.get(opts, :top_k, 5)
    checksum? = Keyword.get(opts, :checksum, true)

    classes = Enum.map(values, &classify/1)
    finite = for {value, :finite} <- Enum.zip(values, classes), do: value * 1.0

    %__MODULE__{
      shape: shape,
      dtype: dtype,
      count: length(values),
      finite_count: length(finite),
      nan_count: Enum.count(classes, &(&1 == :nan)),
      positive_infinity_count: Enum.count(classes, &(&1 == :positive_infinity)),
      negative_infinity_count: Enum.count(classes, &(&1 == :negative_infinity)),
      min: min_or_nil(finite),
      max: max_or_nil(finite),
      mean: mean(finite),
      stddev: stddev(finite),
      l2_norm: l2_norm(finite),
      top_k: top_k(finite, top_k_count),
      entropy: maybe_entropy(finite, opts),
      checksum: maybe_checksum({shape, dtype, values}, checksum?)
    }
  end

  defp infer_shape(values) when is_list(values), do: infer_shape(values, [])

  defp infer_shape([], acc), do: Enum.reverse([0 | acc])

  defp infer_shape([first | _] = values, acc) when is_list(first) do
    child_shape = infer_shape(first, [])

    if Enum.all?(values, &(is_list(&1) and infer_shape(&1, []) == child_shape)) do
      [length(values) | child_shape] |> Enum.reverse(acc)
    else
      [length(values) | Enum.reverse(acc)]
    end
  end

  defp infer_shape(values, acc), do: Enum.reverse([length(values) | acc])

  defp infer_dtype(values) do
    flat = List.flatten(values)

    cond do
      flat == [] -> :unknown
      Enum.all?(flat, &is_integer/1) -> :s64
      Enum.all?(flat, &is_number/1) -> :f64
      true -> :unknown
    end
  end

  defp classify(value) when is_integer(value), do: :finite

  defp classify(value) when is_float(value) do
    case value |> Float.to_string() |> String.downcase() do
      "nan" -> :nan
      "infinity" -> :positive_infinity
      "inf" -> :positive_infinity
      "-infinity" -> :negative_infinity
      "-inf" -> :negative_infinity
      _ -> :finite
    end
  end

  defp classify(_), do: :nan

  defp min_or_nil([]), do: nil
  defp min_or_nil(values), do: Enum.min(values)

  defp max_or_nil([]), do: nil
  defp max_or_nil(values), do: Enum.max(values)

  defp mean([]), do: nil
  defp mean(values), do: Enum.sum(values) / length(values)

  defp stddev([]), do: nil
  defp stddev([_]), do: 0.0

  defp stddev(values) do
    avg = mean(values)

    variance =
      values |> Enum.map(&:math.pow(&1 - avg, 2)) |> Enum.sum() |> Kernel./(length(values))

    :math.sqrt(variance)
  end

  defp l2_norm([]), do: nil

  defp l2_norm(values) do
    values
    |> Enum.map(&(&1 * &1))
    |> Enum.sum()
    |> :math.sqrt()
  end

  defp top_k(_values, count) when count <= 0, do: []

  defp top_k(values, count) do
    values
    |> Enum.sort(:desc)
    |> Enum.take(count)
  end

  defp maybe_entropy(values, opts) do
    if Keyword.get(opts, :entropy, false), do: entropy(values), else: nil
  end

  defp entropy([]), do: nil

  defp entropy(values) do
    max = Enum.max(values)
    exps = Enum.map(values, &:math.exp(&1 - max))
    total = Enum.sum(exps)

    exps
    |> Enum.map(&(&1 / total))
    |> Enum.reject(&(&1 <= 0.0))
    |> Enum.map(&(-&1 * :math.log(&1)))
    |> Enum.sum()
  end

  defp maybe_checksum(_value, false), do: nil

  defp maybe_checksum(value, true) do
    :sha256
    |> :crypto.hash(:erlang.term_to_binary(value))
    |> Base.encode16(case: :lower)
  end
end
