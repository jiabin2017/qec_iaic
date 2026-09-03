from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal, Protocol, Sequence, cast

import numpy as np
from numpy.typing import NDArray
from qldpc.codes import CSSCode, ClassicalCode
from qldpc.decoders import get_decoder_BP_OSD
from qldpc.objects import Pauli


ErrorType = Literal["X", "Z"]
BinaryVector = NDArray[np.uint8]
BinaryMatrix = NDArray[np.uint8]


def validate_probability(p: float) -> float:
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"Physical error rate must satisfy 0 <= p <= 1, got {p}")
    return p


def binary_vector(
    value: Sequence[int] | NDArray[np.integer],
    length: int,
    name: str,
) -> BinaryVector:
    vector = np.asarray(value, dtype=int).reshape(-1)
    if vector.size != length:
        raise ValueError(f"{name} must have length {length}, got {vector.size}")
    if not np.isin(vector, [0, 1]).all():
        raise ValueError(f"{name} must be binary over GF(2)")
    return vector.astype(np.uint8, copy=True)


@dataclass(frozen=True)
class DecodingProblem:
    """Select one binary X- or Z-error channel from a binary CSSCode."""

    code: CSSCode
    error_type: ErrorType | str
    check_matrix: BinaryMatrix = field(init=False, repr=False)
    stabilizer_matrix: BinaryMatrix = field(init=False, repr=False)
    dual_logicals: BinaryMatrix = field(init=False, repr=False)
    check_rank: int = field(init=False)

    def __post_init__(self) -> None:
        error_type = str(self.error_type).upper()
        if error_type not in {"X", "Z"}:
            raise ValueError(f"error_type must be 'X' or 'Z', got {self.error_type!r}")
        object.__setattr__(self, "error_type", cast(ErrorType, error_type))

        if self.code.field.order != 2:
            raise ValueError("This runner currently supports binary CSS codes only")
        if self.code.is_subsystem_code:
            raise ValueError("This runner expects a commuting CSS stabilizer code")

        if error_type == "X":
            check = self.code.matrix_z
            stabilizers = self.code.matrix_x
            dual_logicals = self.code.get_logical_ops(Pauli.Z)
            check_rank = self.code.code_z.rank
        else:
            check = self.code.matrix_x
            stabilizers = self.code.matrix_z
            dual_logicals = self.code.get_logical_ops(Pauli.X)
            check_rank = self.code.code_x.rank

        n = self.code.num_qudits
        check_array = np.asarray(check, dtype=np.uint8).reshape(-1, n).copy()
        stabilizer_array = np.asarray(stabilizers, dtype=np.uint8).reshape(-1, n).copy()
        logical_array = np.asarray(dual_logicals, dtype=np.uint8).reshape(-1, n).copy()
        check_array.setflags(write=False)
        stabilizer_array.setflags(write=False)
        logical_array.setflags(write=False)

        object.__setattr__(self, "check_matrix", check_array)
        object.__setattr__(self, "stabilizer_matrix", stabilizer_array)
        object.__setattr__(self, "dual_logicals", logical_array)
        object.__setattr__(self, "check_rank", int(check_rank))

    @property
    def n(self) -> int:
        return int(self.code.num_qudits)

    @property
    def k(self) -> int:
        return int(self.code.dimension)

    @property
    def num_checks(self) -> int:
        return int(self.check_matrix.shape[0])

    def validate_error(self, error: Sequence[int] | NDArray[np.integer]) -> BinaryVector:
        return binary_vector(error, self.n, "error")

    def validate_syndrome(
        self,
        syndrome: Sequence[int] | NDArray[np.integer],
    ) -> BinaryVector:
        return binary_vector(syndrome, self.num_checks, "syndrome")

    def syndrome(self, error: Sequence[int] | NDArray[np.integer]) -> BinaryVector:
        error_vector = self.validate_error(error)
        return ((self.check_matrix @ error_vector) % 2).astype(np.uint8)

    def syndrome_is_reachable(
        self,
        syndrome: Sequence[int] | NDArray[np.integer],
    ) -> bool:
        syndrome_vector = self.validate_syndrome(syndrome)
        augmented = np.column_stack([self.check_matrix, syndrome_vector])
        augmented_rank = ClassicalCode(augmented, field=2).rank
        return int(augmented_rank) == self.check_rank

    def logical_signature(
        self,
        residual: Sequence[int] | NDArray[np.integer],
    ) -> BinaryVector:
        residual_vector = self.validate_error(residual)
        return ((self.dual_logicals @ residual_vector) % 2).astype(np.uint8)


class NoiseModel(Protocol):
    name: str
    error_type: ErrorType

    def sample(
        self,
        n: int,
        p: float,
        shots: int,
        rng: np.random.Generator,
    ) -> BinaryMatrix: ...

    def probability(self, error: BinaryVector, p: float) -> float: ...


def sample_binary_iid(
    n: int,
    p: float,
    shots: int,
    rng: np.random.Generator,
) -> BinaryMatrix:
    validate_probability(p)
    if shots <= 0:
        raise ValueError(f"shots must be positive, got {shots}")
    return (rng.random((shots, n)) < p).astype(np.uint8)


def binary_iid_probability(error: BinaryVector, p: float) -> float:
    p = validate_probability(p)
    weight = int(np.count_nonzero(error))
    return float((p**weight) * ((1.0 - p) ** (error.size - weight)))


@dataclass(frozen=True)
class BitFlipNoise:
    name: ClassVar[str] = "bit_flip"
    error_type: ClassVar[ErrorType] = "X"

    def sample(
        self,
        n: int,
        p: float,
        shots: int,
        rng: np.random.Generator,
    ) -> BinaryMatrix:
        return sample_binary_iid(n, p, shots, rng)

    def probability(self, error: BinaryVector, p: float) -> float:
        return binary_iid_probability(error, p)


@dataclass(frozen=True)
class PhaseFlipNoise:
    name: ClassVar[str] = "phase_flip"
    error_type: ClassVar[ErrorType] = "Z"

    def sample(
        self,
        n: int,
        p: float,
        shots: int,
        rng: np.random.Generator,
    ) -> BinaryMatrix:
        return sample_binary_iid(n, p, shots, rng)

    def probability(self, error: BinaryVector, p: float) -> float:
        return binary_iid_probability(error, p)


class SyndromeDecoder(Protocol):
    def decode(self, syndrome: NDArray[np.integer]) -> NDArray[np.integer]: ...


class DecoderFactory(Protocol):
    name: str

    def build(self, problem: DecodingProblem, p: float) -> SyndromeDecoder: ...


@dataclass(frozen=True)
class BPOSDDecoderFactory:
    """Create one qldpc BP+OSD decoder for each physical error rate."""

    max_iter: int = 50
    bp_method: str = "minimum_sum"
    schedule: str = "parallel"
    osd_method: str = "OSD_CS"
    osd_order: int = 2
    name: ClassVar[str] = "bp_osd"

    def __post_init__(self) -> None:
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.osd_method == "OSD_0" and self.osd_order != 0:
            raise ValueError("OSD_0 requires osd_order=0")

    def build(self, problem: DecodingProblem, p: float) -> SyndromeDecoder:
        p = validate_probability(p)
        if p in {0.0, 1.0}:
            raise ValueError("BP+OSD requires 0 < p < 1")
        return get_decoder_BP_OSD(
            problem.check_matrix,
            error_rate=p,
            max_iter=self.max_iter,
            bp_method=self.bp_method,
            schedule=self.schedule,
            input_vector_type="syndrome",
            osd_method=self.osd_method,
            osd_order=self.osd_order,
        )


@dataclass(frozen=True)
class DecodeResult:
    p: float
    syndrome: BinaryVector
    correction: BinaryVector
    syndrome_valid: bool
    converged: bool | None
    iterations: int | None


@dataclass(frozen=True)
class TrialResult:
    p: float
    true_error: BinaryVector
    true_error_probability: float
    syndrome: BinaryVector
    correction: BinaryVector
    residual: BinaryVector
    logical_signature: BinaryVector
    syndrome_valid: bool
    logical_failure: bool
    total_failure: bool
    converged: bool | None
    iterations: int | None


@dataclass(frozen=True)
class SweepPoint:
    p: float
    shots: int
    failures: int
    syndrome_invalid_failures: int
    logical_failures: int
    logical_error_rate: float
    logical_error_rate_stderr: float
    mean_error_weight: float
    mean_correction_weight: float
    mean_residual_weight: float
    decoder_converged_count: int | None
    decoder_converged_rate: float | None
    mean_decoder_iterations: float | None


@dataclass(frozen=True)
class SweepResult:
    code_name: str
    n: int
    k: int
    error_type: ErrorType
    noise_model: str
    decoder_name: str
    seed: int
    points: tuple[SweepPoint, ...]

    def write_csv(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "code",
            "n",
            "k",
            "error_type",
            "noise_model",
            "decoder",
            "seed",
            "p",
            "shots",
            "failures",
            "syndrome_invalid_failures",
            "logical_failures",
            "logical_error_rate",
            "logical_error_rate_stderr",
            "mean_error_weight",
            "mean_correction_weight",
            "mean_residual_weight",
            "decoder_converged_count",
            "decoder_converged_rate",
            "mean_decoder_iterations",
        ]
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for point in self.points:
                writer.writerow(
                    {
                        "code": self.code_name,
                        "n": self.n,
                        "k": self.k,
                        "error_type": self.error_type,
                        "noise_model": self.noise_model,
                        "decoder": self.decoder_name,
                        "seed": self.seed,
                        "p": point.p,
                        "shots": point.shots,
                        "failures": point.failures,
                        "syndrome_invalid_failures": point.syndrome_invalid_failures,
                        "logical_failures": point.logical_failures,
                        "logical_error_rate": point.logical_error_rate,
                        "logical_error_rate_stderr": point.logical_error_rate_stderr,
                        "mean_error_weight": point.mean_error_weight,
                        "mean_correction_weight": point.mean_correction_weight,
                        "mean_residual_weight": point.mean_residual_weight,
                        "decoder_converged_count": point.decoder_converged_count,
                        "decoder_converged_rate": point.decoder_converged_rate,
                        "mean_decoder_iterations": point.mean_decoder_iterations,
                    }
                )
        return output_path


@dataclass
class CodeCapacityRunner:
    problem: DecodingProblem
    noise: NoiseModel
    decoder_factory: DecoderFactory
    seed: int = 1234
    code_name: str | None = None

    def __post_init__(self) -> None:
        if self.noise.error_type != self.problem.error_type:
            raise ValueError(
                f"{self.noise.name} produces {self.noise.error_type} errors, but the "
                f"selected problem decodes {self.problem.error_type} errors"
            )
        if self.code_name is None:
            self.code_name = type(self.problem.code).__name__

    def _decode_with(
        self,
        decoder: SyndromeDecoder,
        syndrome: BinaryVector,
        p: float,
    ) -> DecodeResult:
        correction = self.problem.validate_error(decoder.decode(syndrome))
        produced_syndrome = self.problem.syndrome(correction)
        syndrome_valid = bool(np.array_equal(produced_syndrome, syndrome))

        converged_value = getattr(decoder, "converge", None)
        iterations_value = getattr(decoder, "iter", None)
        converged = None if converged_value is None else bool(converged_value)
        iterations = None if iterations_value is None else int(iterations_value)

        return DecodeResult(
            p=p,
            syndrome=syndrome.copy(),
            correction=correction,
            syndrome_valid=syndrome_valid,
            converged=converged,
            iterations=iterations,
        )

    def _evaluate_error_with(
        self,
        decoder: SyndromeDecoder,
        true_error: BinaryVector,
        p: float,
    ) -> TrialResult:
        syndrome = self.problem.syndrome(true_error)
        decoded = self._decode_with(decoder, syndrome, p)
        residual = np.bitwise_xor(true_error, decoded.correction).astype(np.uint8)
        logical_signature = self.problem.logical_signature(residual)
        logical_failure = decoded.syndrome_valid and bool(np.any(logical_signature))
        total_failure = (not decoded.syndrome_valid) or logical_failure

        return TrialResult(
            p=p,
            true_error=true_error.copy(),
            true_error_probability=self.noise.probability(true_error, p),
            syndrome=syndrome,
            correction=decoded.correction,
            residual=residual,
            logical_signature=logical_signature,
            syndrome_valid=decoded.syndrome_valid,
            logical_failure=logical_failure,
            total_failure=total_failure,
            converged=decoded.converged,
            iterations=decoded.iterations,
        )

    def run_shots(
        self,
        p: float,
        shots: int,
        rng: np.random.Generator,
    ) -> SweepPoint:
        p = validate_probability(p)
        if shots <= 0:
            raise ValueError(f"shots must be positive, got {shots}")

        decoder = self.decoder_factory.build(self.problem, p)
        errors = self.noise.sample(self.problem.n, p, shots, rng)
        trials = [
            self._evaluate_error_with(decoder, errors[index], p) for index in range(shots)
        ]

        failures = sum(trial.total_failure for trial in trials)
        syndrome_invalid = sum(not trial.syndrome_valid for trial in trials)
        logical_failures = sum(trial.logical_failure for trial in trials)
        logical_error_rate = failures / shots
        stderr = float(np.sqrt(logical_error_rate * (1.0 - logical_error_rate) / shots))

        converged_values = [
            trial.converged for trial in trials if trial.converged is not None
        ]
        iteration_values = [
            trial.iterations for trial in trials if trial.iterations is not None
        ]
        converged_count = sum(converged_values) if converged_values else None
        converged_rate = (
            converged_count / len(converged_values) if converged_count is not None else None
        )
        mean_iterations = float(np.mean(iteration_values)) if iteration_values else None

        return SweepPoint(
            p=p,
            shots=shots,
            failures=failures,
            syndrome_invalid_failures=syndrome_invalid,
            logical_failures=logical_failures,
            logical_error_rate=logical_error_rate,
            logical_error_rate_stderr=stderr,
            mean_error_weight=float(np.mean(errors.sum(axis=1))),
            mean_correction_weight=float(np.mean([trial.correction.sum() for trial in trials])),
            mean_residual_weight=float(np.mean([trial.residual.sum() for trial in trials])),
            decoder_converged_count=converged_count,
            decoder_converged_rate=converged_rate,
            mean_decoder_iterations=mean_iterations,
        )

    def sweep(self, p_values: Sequence[float], shots: int = 10_000) -> SweepResult:
        probabilities = [validate_probability(p) for p in p_values]
        if not probabilities:
            raise ValueError("p_values must not be empty")
        if shots <= 0:
            raise ValueError(f"shots must be positive, got {shots}")

        child_seeds = np.random.SeedSequence(self.seed).spawn(len(probabilities))
        points = tuple(
            self.run_shots(p, shots, np.random.default_rng(child_seed))
            for p, child_seed in zip(probabilities, child_seeds, strict=True)
        )
        assert self.code_name is not None
        return SweepResult(
            code_name=self.code_name,
            n=self.problem.n,
            k=self.problem.k,
            error_type=cast(ErrorType, self.problem.error_type),
            noise_model=self.noise.name,
            decoder_name=self.decoder_factory.name,
            seed=self.seed,
            points=points,
        )
