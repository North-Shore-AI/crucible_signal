defmodule CrucibleSignal do
  @moduledoc """
  Canonical signal ontology for transformer forward-pass artifacts.

  This package owns reusable contracts for signal names, references,
  capabilities, tensor metadata, and capture posture. It intentionally avoids
  model loading, orchestration, and product-specific routing.
  """

  @version Mix.Project.config()[:version]

  @doc "Returns the package version."
  def version, do: @version
end
