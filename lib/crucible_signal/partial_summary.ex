defmodule CrucibleSignal.PartialSummary do
  @moduledoc """
  Count-compatible accumulator for streaming tensor statistics.
  """

  alias CrucibleSignal.TensorSummary

  @derive Jason.Encoder
  defstruct dtype: :unknown,
            count: 0,
            finite_count: 0,
            nan_count: 0,
            positive_infinity_count: 0,
            negative_infinity_count: 0,
            min: nil,
            max: nil,
            sum: 0.0,
            sum_sq: 0.0,
            l2_norm_sq: 0.0

  @type t :: %__MODULE__{}

  def from_values(values, dtype \\ :unknown) when is_list(values) do
    Enum.reduce(List.flatten(values), %__MODULE__{dtype: dtype}, &accumulate/2)
  end

  def from_summary(%TensorSummary{} = summary) do
    mean = summary.mean || 0.0
    finite_count = summary.finite_count || 0
    stddev = summary.stddev || 0.0
    variance = stddev * stddev
    sum = mean * finite_count
    sum_sq = variance * finite_count + finite_count * mean * mean
    l2_norm = summary.l2_norm || 0.0

    %__MODULE__{
      dtype: summary.dtype,
      count: summary.count,
      finite_count: finite_count,
      nan_count: summary.nan_count,
      positive_infinity_count: summary.positive_infinity_count,
      negative_infinity_count: summary.negative_infinity_count,
      min: summary.min,
      max: summary.max,
      sum: sum,
      sum_sq: sum_sq,
      l2_norm_sq: l2_norm * l2_norm
    }
  end

  def merge(%__MODULE__{} = left, %__MODULE__{} = right) do
    %__MODULE__{
      dtype: if(left.dtype == :unknown, do: right.dtype, else: left.dtype),
      count: left.count + right.count,
      finite_count: left.finite_count + right.finite_count,
      nan_count: left.nan_count + right.nan_count,
      positive_infinity_count: left.positive_infinity_count + right.positive_infinity_count,
      negative_infinity_count: left.negative_infinity_count + right.negative_infinity_count,
      min: min_ignore_nil(left.min, right.min),
      max: max_ignore_nil(left.max, right.max),
      sum: left.sum + right.sum,
      sum_sq: left.sum_sq + right.sum_sq,
      l2_norm_sq: left.l2_norm_sq + right.l2_norm_sq
    }
  end

  def to_summary(%__MODULE__{} = partial) do
    mean = if partial.finite_count == 0, do: nil, else: partial.sum / partial.finite_count

    stddev =
      if partial.finite_count == 0 do
        nil
      else
        variance = max(partial.sum_sq / partial.finite_count - mean * mean, 0.0)
        :math.sqrt(variance)
      end

    %TensorSummary{
      dtype: partial.dtype,
      count: partial.count,
      finite_count: partial.finite_count,
      nan_count: partial.nan_count,
      positive_infinity_count: partial.positive_infinity_count,
      negative_infinity_count: partial.negative_infinity_count,
      min: partial.min,
      max: partial.max,
      mean: mean,
      stddev: stddev,
      l2_norm: :math.sqrt(partial.l2_norm_sq),
      top_k: [],
      entropy: nil,
      checksum: nil
    }
  end

  defp accumulate(value, acc) when is_number(value) do
    value = value * 1.0

    %{
      acc
      | count: acc.count + 1,
        finite_count: acc.finite_count + 1,
        min: min_ignore_nil(acc.min, value),
        max: max_ignore_nil(acc.max, value),
        sum: acc.sum + value,
        sum_sq: acc.sum_sq + value * value,
        l2_norm_sq: acc.l2_norm_sq + value * value
    }
  end

  defp accumulate(_value, acc), do: %{acc | count: acc.count + 1, nan_count: acc.nan_count + 1}

  defp min_ignore_nil(nil, value), do: value
  defp min_ignore_nil(value, nil), do: value
  defp min_ignore_nil(left, right), do: min(left, right)

  defp max_ignore_nil(nil, value), do: value
  defp max_ignore_nil(value, nil), do: value
  defp max_ignore_nil(left, right), do: max(left, right)
end
