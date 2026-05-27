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

  @doc "Returns all canonical signal types."
  def signal_types, do: CrucibleSignal.SignalType.all()

  @doc "Returns all canonical operation types."
  def operations, do: CrucibleSignal.Operation.all()

  @doc "Returns all supported capture modes."
  def capture_modes, do: CrucibleSignal.CaptureMode.all()
end
