from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class MethodContract:
    """Machine-readable comparability and online-input contract."""

    method: str
    display_name: str
    privacy_definition: str
    online_inputs: tuple[str, ...]
    uses_sensitive_value_online: bool
    deployable_under_capt: bool
    output_bits: int
    single_output: bool
    formal_comparable: bool
    role: str
    source_kind: str
    exclusion_reason: str | None = None

    def __post_init__(self) -> None:
        if self.output_bits != 12 or not self.single_output:
            if self.formal_comparable:
                raise ValueError("formal CAPT comparison requires one 12-bit output")
        if self.uses_sensitive_value_online and self.deployable_under_capt:
            raise ValueError("a method using A_S online is not deployable under CAPT")
        if self.role not in {"main", "oracle", "diagnostic", "reject"}:
            raise ValueError("invalid comparison role")
        if self.role == "reject" and not self.exclusion_reason:
            raise ValueError("rejected methods require an exclusion reason")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_method_contracts() -> list[MethodContract]:
    """Return contracts in deterministic publication order."""
    return [
        MethodContract(
            "capt",
            "CAPT-12",
            "robust profile privacy",
            ("Z", "B", "S"),
            False,
            True,
            12,
            True,
            True,
            "main",
            "native",
        ),
        MethodContract(
            "optimal_ldp",
            "optimal LDP",
            "epsilon-local differential privacy",
            ("Z", "B", "S"),
            False,
            True,
            12,
            True,
            True,
            "main",
            "native",
        ),
        MethodContract(
            "best_constant_cover",
            "best constant-row cover",
            "input-independent (epsilon=0)",
            ("B", "S"),
            False,
            True,
            12,
            True,
            True,
            "main",
            "native",
        ),
        MethodContract(
            "pbp_oracle",
            "Profile-Based Privacy oracle",
            "nominal profile-based privacy",
            ("Z", "B", "S", "A_S"),
            True,
            False,
            12,
            True,
            False,
            "oracle",
            "paper-derived LP",
            "requires the realized sensitive profile online",
        ),
        MethodContract(
            "pbp_common_nominal",
            "common-channel nominal PBP",
            "point-estimate profile privacy",
            ("Z", "B", "S"),
            False,
            True,
            12,
            True,
            False,
            "diagnostic",
            "paper-derived LP",
            "formal only after the frozen channel passes the robust verifier",
        ),
        MethodContract(
            "pbp_common_nominal_calibrated",
            "common-channel nominal PBP + certified cover calibration",
            "robust profile privacy after common wrapper",
            ("Z", "B", "S"),
            False,
            True,
            12,
            True,
            True,
            "main",
            "CAPT finite-output adaptation",
        ),
        MethodContract(
            "mass12_raw",
            "MaSS-12 (finite-output adaptation)",
            "information-theoretic penalty; uncertified until verified",
            ("Z", "B", "S"),
            False,
            True,
            12,
            True,
            False,
            "diagnostic",
            "finite-output adaptation",
            "MaSS MI controls do not imply the CAPT robust certificate by themselves",
        ),
        MethodContract(
            "mass12_calibrated",
            "MaSS-12 + certified cover calibration",
            "robust profile privacy after common wrapper",
            ("Z", "B", "S"),
            False,
            True,
            12,
            True,
            True,
            "main",
            "finite-output adaptation plus CAPT wrapper",
        ),
        MethodContract(
            "pufferfish_framework",
            "Pufferfish framework",
            "Pufferfish privacy",
            (),
            False,
            False,
            0,
            False,
            False,
            "reject",
            "framework",
            "privacy definition, not a concrete single-token mechanism",
        ),
        MethodContract(
            "pufferfish_wasserstein",
            "Pufferfish Wasserstein mechanism",
            "Pufferfish privacy for central query release",
            (),
            False,
            False,
            0,
            False,
            False,
            "reject",
            "paper mechanism",
            "central numeric-query mechanism does not implement the local token interface",
        ),
        MethodContract(
            "pufferfish_markov_quilt",
            "Pufferfish Markov Quilt mechanism",
            "Pufferfish privacy for correlated time series",
            (),
            False,
            False,
            0,
            False,
            False,
            "reject",
            "paper mechanism",
            "requires a Markov model and central query rather than one local 12-bit token",
        ),
        MethodContract(
            "distp_tupling",
            "Distribution Privacy tupling mechanism",
            "distribution privacy",
            ("Z",),
            False,
            False,
            12,
            False,
            False,
            "reject",
            "paper mechanism",
            "emits a tuple containing the real and dummy outputs, not one 12-bit token",
        ),
    ]
