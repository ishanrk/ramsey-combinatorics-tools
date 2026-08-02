from __future__ import annotations

from pvdw.proof_tools.block_cover import (
    Block,
    check_one_exception_signature_obstruction,
    enumerate_translated_blocks,
    find_exact_cover,
    find_lexicographic_exact_cover,
    spindle_sevenfold_blocks,
    verify_block_cover_witness,
    verify_uniform_cover,
)


def test_spindle_seventeen_block_sevenfold_regression() -> None:
    blocks = spindle_sevenfold_blocks()
    assert len(blocks) == 17
    assert verify_uniform_cover(blocks, range(17), multiplicity=7)
    assert [(block.shape_name, block.translation) for block in blocks] == [
        ("A", 0), ("A", 1),
        ("B", 0), ("B", 1), ("B", 2),
        ("C", 0), ("C", 4),
        ("D0", 0), ("D0", 3), ("D0", 4),
        ("E", 0), ("E", 1), ("E", 4),
        ("F", 0), ("F", 4),
        ("G", 0), ("G", 1),
    ]


def test_modular_one_exception_obstruction() -> None:
    result = check_one_exception_signature_obstruction()
    assert result.residue == 6
    assert result.permitted_counts == (6, 13)
    assert result.minimum_total == 24
    assert result.impossible
    assert result.explanation == (
        "n_j ≡ 6 mod 7",
        "n_j ∈ {6,13}",
        "sum_j n_j >= 24 > 17",
    )


def test_cp_sat_exact_cover_and_multiplicity_range() -> None:
    blocks = enumerate_translated_blocks(
        {"pair": (0, 1)}, 0, 3, include_reflections=True
    )
    witness = find_exact_cover(
        blocks,
        range(4),
        multiplicity=1,
        multiplicity_range=None,
        minimize_block_count=True,
    )
    assert witness.verified
    assert verify_block_cover_witness(witness)
    assert len(witness.selected_blocks) == 2
    assert verify_uniform_cover(witness.selected_blocks, range(4), 1)

    repeated = find_exact_cover(
        (Block((0,), "point", 0, False),),
        (0,),
        multiplicity=None,
        multiplicity_range=(2, 3),
        binary_selection=False,
    )
    assert repeated.multiplicity == 2
    assert len(repeated.selected_blocks) == 2

    lexicographic = find_lexicographic_exact_cover(
        {"pair": (0, 1)},
        0,
        5,
        multiplicity=1,
        multiplicity_range=None,
    )
    assert lexicographic.ambient_points == (0, 1)
    assert len(lexicographic.selected_blocks) == 1
