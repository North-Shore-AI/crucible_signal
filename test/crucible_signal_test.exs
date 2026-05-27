defmodule CrucibleSignalTest do
  use ExUnit.Case
  doctest CrucibleSignal

  test "exposes package version" do
    assert CrucibleSignal.version() == "0.1.0"
  end
end
