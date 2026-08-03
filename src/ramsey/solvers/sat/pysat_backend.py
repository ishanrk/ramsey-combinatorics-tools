from __future__ import annotations

from time import perf_counter

from ramsey.solvers.sat.formula import CNFFormula
from ramsey.solvers.sat.model import SatResult, SatStatus

PYSAT_BACKENDS = {
    "pysat": "m22",
    "cadical": "cadical195",
    "glucose": "g4",
    "glucose3": "g3",
    "minisat": "m22",
    "minisat22": "m22",
}


def pysat_available() -> bool:
    try:
        import pysat.solvers
    except ImportError:
        return False
    return True


def solve_with_pysat(
    formula: CNFFormula,
    *,
    backend: str,
    assumptions: tuple[int, ...] = (),
) -> SatResult:
    """solve through python-sat while keeping imports optional."""
    started = perf_counter()
    try:
        from pysat.solvers import Solver
    except ImportError:
        return SatResult(
            status=SatStatus.ERROR,
            backend=backend,
            model=None,
            elapsed_seconds=perf_counter() - started,
            message="python-sat is not installed",
        )

    solver_name = PYSAT_BACKENDS[backend]
    try:
        with Solver(name=solver_name, bootstrap_with=formula.clauses) as solver:
            satisfiable = solver.solve(assumptions=list(assumptions))
            if not satisfiable:
                return SatResult(
                    status=SatStatus.UNSAT,
                    backend=backend,
                    model=None,
                    elapsed_seconds=perf_counter() - started,
                )
            raw_model = solver.get_model() or []
    except Exception as exc:
        return SatResult(
            status=SatStatus.ERROR,
            backend=backend,
            model=None,
            elapsed_seconds=perf_counter() - started,
            message=str(exc),
        )

    values = {abs(literal): literal > 0 for literal in raw_model}
    model = tuple(
        variable if values.get(variable, False) else -variable
        for variable in range(1, formula.variable_count + 1)
    )
    return SatResult(
        status=SatStatus.SAT,
        backend=backend,
        model=model,
        elapsed_seconds=perf_counter() - started,
    )


__all__ = ["PYSAT_BACKENDS", "pysat_available", "solve_with_pysat"]
