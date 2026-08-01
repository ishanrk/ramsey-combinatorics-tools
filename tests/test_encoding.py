from __future__ import annotations

import itertools

import pytest
from hypothesis import given, settings, strategies as st
from pysat.solvers import Solver

from pvdw.distances import generate_distances
from pvdw.encoding.binary import decode_binary_model, encode_binary
from pvdw.encoding.common import BinaryVariableMap, ListClauseSink, OneHotVariableMap
from pvdw.encoding.onehot import AtMostOneEncoding, encode_onehot
from pvdw.graph import DistanceGraph
from pvdw.model import InstanceSpec, PolynomialSpec
from pvdw.verify import verify_coloring


def satisfies(clauses: list[list[int]], true_variables: set[int]) -> bool:
    return all(
        any((literal > 0) == (abs(literal) in true_variables) for literal in clause)
        for clause in clauses
    )


def graph_for(instance: InstanceSpec) -> DistanceGraph:
    return DistanceGraph(instance.n, generate_distances(instance).values)


@st.composite
def tiny_instances(draw: st.DrawFn) -> InstanceSpec:
    degree = draw(st.integers(min_value=1, max_value=3))
    coefficients = [0]
    if degree > 1:
        coefficients.extend(
            draw(
                st.lists(
                    st.integers(min_value=-2, max_value=2),
                    min_size=degree - 1,
                    max_size=degree - 1,
                )
            )
        )
    coefficients.append(draw(st.sampled_from([-2, -1, 1, 2])))
    return InstanceSpec(
        PolynomialSpec(tuple(coefficients)),
        colors=draw(st.integers(min_value=2, max_value=4)),
        n=draw(st.integers(min_value=1, max_value=5)),
    )


@given(tiny_instances())
@settings(max_examples=30, deadline=None)
def test_onehot_and_binary_agree_with_every_coloring(instance: InstanceSpec) -> None:
    graph = graph_for(instance)
    onehot_sink = ListClauseSink()
    onehot = encode_onehot(
        instance,
        graph,
        onehot_sink,
        amo=AtMostOneEncoding.PAIRWISE,
        fix_first_color=False,
    )
    binary_sink = ListClauseSink()
    binary = encode_binary(instance, graph, binary_sink, fix_first_color=False)
    assert onehot.statistics.clauses == len(onehot_sink.clauses)
    assert binary.statistics.clauses == len(binary_sink.clauses)

    onehot_map = OneHotVariableMap(instance.n, instance.colors)
    binary_map = BinaryVariableMap(instance.n, (instance.colors - 1).bit_length())
    for coloring in itertools.product(range(instance.colors), repeat=instance.n):
        valid = verify_coloring(instance, coloring).valid
        onehot_true = {
            onehot_map.var(vertex, color)
            for vertex, color in enumerate(coloring)
        }
        binary_true = {
            binary_map.var(vertex, bit)
            for vertex, color in enumerate(coloring)
            for bit in range(binary_map.bits)
            if (color >> bit) & 1
        }
        assert satisfies(onehot_sink.clauses, onehot_true) is valid
        assert satisfies(binary_sink.clauses, binary_true) is valid


@pytest.mark.parametrize(
    "amo",
    [
        AtMostOneEncoding.SEQUENTIAL,
        AtMostOneEncoding.LADDER,
        AtMostOneEncoding.BITWISE,
    ],
)
def test_auxiliary_at_most_one_encodings_and_accounting(
    amo: AtMostOneEncoding,
) -> None:
    instance = InstanceSpec(PolynomialSpec((0, 0, 1)), 4, 5)
    graph = graph_for(instance)
    sink = ListClauseSink()
    result = encode_onehot(
        instance, graph, sink, amo=amo, fix_first_color=False
    )
    assert result.statistics.clauses == len(sink.clauses)
    assert result.statistics.literals == sum(map(len, sink.clauses))
    assert result.statistics.variables >= instance.n * instance.colors
    with Solver(bootstrap_with=sink.clauses) as solver:
        variables = OneHotVariableMap(instance.n, instance.colors)
        for coloring in itertools.product(range(instance.colors), repeat=instance.n):
            assumptions = [
                variables.var(vertex, candidate)
                if candidate == color
                else -variables.var(vertex, candidate)
                for vertex, color in enumerate(coloring)
                for candidate in range(instance.colors)
            ]
            assert solver.solve(assumptions=assumptions) is verify_coloring(
                instance, coloring
            ).valid


def test_clique_obstruction_emits_immediate_unsat() -> None:
    instance = InstanceSpec(PolynomialSpec((0, 1)), 2, 3)
    sink = ListClauseSink()
    result = encode_onehot(
        instance,
        graph_for(instance),
        sink,
        amo=AtMostOneEncoding.PAIRWISE,
        anchor_clique=True,
    )
    assert result.immediate_noncolorability
    assert sink.clauses == [[]]


def test_decode_binary_model_and_rejections() -> None:
    instance = InstanceSpec(PolynomialSpec((0, 1)), 3, 2)
    # Codes 2 and 1 with low bit first: v0=(0,1), v1=(1,0).
    assert decode_binary_model([-1, 2, 3, -4], instance) == (2, 1)
    for malformed in (
        [-1, 2, 3],
        [-1, 1, 2, 3, -4],
        [-1, 2, 3, -4, 5],
        [0, 2, 3, -4],
        [1, 2, 3, 4],  # invalid code 3 at both vertices
    ):
        with pytest.raises(ValueError):
            decode_binary_model(malformed, instance)
