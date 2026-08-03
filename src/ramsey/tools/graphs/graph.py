from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence


def _iter_bits(mask: int) -> Iterator[int]:
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


class Graph:
    """a compact mutable simple graph backed by python integer bitsets."""

    __slots__ = ("_adjacency", "_edge_count", "_vertex_count")

    def __init__(
        self,
        vertex_count: int,
        edges: Iterable[tuple[int, int]] = (),
    ) -> None:
        self._validate_vertex_count(vertex_count)
        self._vertex_count = vertex_count
        self._adjacency = [0] * vertex_count
        self._edge_count = 0
        for edge in edges:
            try:
                u, v = edge
            except (TypeError, ValueError) as exc:
                raise ValueError("each edge must have two vertices") from exc
            self.add_edge(u, v)

    @classmethod
    def empty(cls, vertex_count: int) -> Graph:
        """make an empty graph on the requested vertices."""
        return cls(vertex_count)

    @classmethod
    def complete(cls, vertex_count: int) -> Graph:
        """make a complete graph without walking through every edge."""
        cls._validate_vertex_count(vertex_count)
        all_vertices = (1 << vertex_count) - 1
        masks = [all_vertices ^ (1 << vertex) for vertex in range(vertex_count)]
        return cls._from_masks(masks, vertex_count * (vertex_count - 1) // 2)

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[tuple[int, int]],
        *,
        vertex_count: int | None = None,
    ) -> Graph:
        """make a graph and infer its order when that is handy."""
        saved_edges = tuple(edges)
        if vertex_count is None:
            largest = -1
            for edge in saved_edges:
                try:
                    u, v = edge
                except (TypeError, ValueError) as exc:
                    raise ValueError("each edge must have two vertices") from exc
                if not isinstance(u, int) or isinstance(u, bool):
                    raise TypeError("vertices must be ordinary integers")
                if not isinstance(v, int) or isinstance(v, bool):
                    raise TypeError("vertices must be ordinary integers")
                if u < 0 or v < 0:
                    raise ValueError("vertices must be nonnegative")
                largest = max(largest, u, v)
            vertex_count = largest + 1
        return cls(vertex_count, saved_edges)

    @classmethod
    def from_adjacency_masks(cls, masks: Sequence[int]) -> Graph:
        """make a graph from symmetric loop-free adjacency masks."""
        vertex_count = len(masks)
        limit = (1 << vertex_count) - 1
        saved_masks = list(masks)
        for vertex, mask in enumerate(saved_masks):
            if not isinstance(mask, int) or isinstance(mask, bool):
                raise TypeError("adjacency masks must be ordinary integers")
            if mask < 0 or mask & ~limit:
                raise ValueError("an adjacency mask contains an unknown vertex")
            if mask & (1 << vertex):
                raise ValueError("self-loops are not allowed")
        for vertex, mask in enumerate(saved_masks):
            for neighbor in _iter_bits(mask):
                if not saved_masks[neighbor] & (1 << vertex):
                    raise ValueError("adjacency masks must be symmetric")
        edge_count = sum(mask.bit_count() for mask in saved_masks) // 2
        return cls._from_masks(saved_masks, edge_count)

    @classmethod
    def _from_masks(cls, masks: list[int], edge_count: int) -> Graph:
        graph = cls.__new__(cls)
        graph._vertex_count = len(masks)
        graph._adjacency = masks
        graph._edge_count = edge_count
        return graph

    @staticmethod
    def _validate_vertex_count(vertex_count: int) -> None:
        if not isinstance(vertex_count, int) or isinstance(vertex_count, bool):
            raise TypeError("vertex_count must be an ordinary integer")
        if vertex_count < 0:
            raise ValueError("vertex_count must be nonnegative")

    def _check_vertex(self, vertex: int) -> None:
        if not isinstance(vertex, int) or isinstance(vertex, bool):
            raise TypeError("vertices must be ordinary integers")
        if not 0 <= vertex < self._vertex_count:
            raise IndexError(f"vertex {vertex} is outside this graph")

    @property
    def vertex_count(self) -> int:
        return self._vertex_count

    @property
    def order(self) -> int:
        return self._vertex_count

    @property
    def edge_count(self) -> int:
        return self._edge_count

    @property
    def size(self) -> int:
        return self._edge_count

    @property
    def adjacency_masks(self) -> tuple[int, ...]:
        return tuple(self._adjacency)

    def __len__(self) -> int:
        return self._vertex_count

    def __iter__(self) -> Iterator[int]:
        return iter(range(self._vertex_count))

    def __contains__(self, vertex: object) -> bool:
        return (
            isinstance(vertex, int)
            and not isinstance(vertex, bool)
            and 0 <= vertex < self._vertex_count
        )

    def add_vertices(self, count: int = 1) -> range:
        """append vertices and return the new range."""
        self._validate_vertex_count(count)
        start = self._vertex_count
        self._vertex_count += count
        self._adjacency.extend([0] * count)
        return range(start, self._vertex_count)

    def add_edge(self, u: int, v: int) -> bool:
        """add one edge and say whether it was new."""
        self._check_vertex(u)
        self._check_vertex(v)
        if u == v:
            raise ValueError("self-loops are not allowed")
        v_bit = 1 << v
        if self._adjacency[u] & v_bit:
            return False
        self._adjacency[u] |= v_bit
        self._adjacency[v] |= 1 << u
        self._edge_count += 1
        return True

    def remove_edge(self, u: int, v: int) -> bool:
        """remove one edge and say whether it was there."""
        self._check_vertex(u)
        self._check_vertex(v)
        if u == v:
            return False
        v_bit = 1 << v
        if not self._adjacency[u] & v_bit:
            return False
        self._adjacency[u] ^= v_bit
        self._adjacency[v] ^= 1 << u
        self._edge_count -= 1
        return True

    def clear_edges(self) -> None:
        """drop every edge while keeping the vertices."""
        self._adjacency = [0] * self._vertex_count
        self._edge_count = 0

    def has_edge(self, u: int, v: int) -> bool:
        self._check_vertex(u)
        self._check_vertex(v)
        return bool(self._adjacency[u] & (1 << v))

    def neighbor_mask(self, vertex: int) -> int:
        self._check_vertex(vertex)
        return self._adjacency[vertex]

    def neighbors(self, vertex: int) -> Iterator[int]:
        self._check_vertex(vertex)
        return _iter_bits(self._adjacency[vertex])

    def degree(self, vertex: int) -> int:
        self._check_vertex(vertex)
        return self._adjacency[vertex].bit_count()

    def degree_sequence(self, *, reverse: bool = True) -> tuple[int, ...]:
        return tuple(
            sorted((mask.bit_count() for mask in self._adjacency), reverse=reverse)
        )

    def iter_edges(self) -> Iterator[tuple[int, int]]:
        for u, mask in enumerate(self._adjacency):
            mask &= -1 << (u + 1)
            for v in _iter_bits(mask):
                yield u, v

    def connected_components(self) -> tuple[tuple[int, ...], ...]:
        """return components using bitset frontiers."""
        unseen = (1 << self._vertex_count) - 1
        components: list[tuple[int, ...]] = []
        while unseen:
            frontier = unseen & -unseen
            unseen ^= frontier
            component = 0
            while frontier:
                component |= frontier
                neighbors = 0
                pending = frontier
                while pending:
                    bit = pending & -pending
                    neighbors |= self._adjacency[bit.bit_length() - 1]
                    pending ^= bit
                frontier = neighbors & unseen
                unseen &= ~frontier
            components.append(tuple(_iter_bits(component)))
        return tuple(components)

    def is_connected(self) -> bool:
        return self._vertex_count <= 1 or len(self.connected_components()) == 1

    def copy(self) -> Graph:
        return self._from_masks(self._adjacency.copy(), self._edge_count)

    def complement(self) -> Graph:
        all_vertices = (1 << self._vertex_count) - 1
        masks = [
            all_vertices ^ mask ^ (1 << vertex)
            for vertex, mask in enumerate(self._adjacency)
        ]
        possible_edges = self._vertex_count * (self._vertex_count - 1) // 2
        return self._from_masks(masks, possible_edges - self._edge_count)

    def induced_subgraph(self, vertices: Iterable[int]) -> Graph:
        """keep the supplied vertices in their supplied order."""
        saved_vertices = tuple(vertices)
        old_to_new: dict[int, int] = {}
        for new, old in enumerate(saved_vertices):
            self._check_vertex(old)
            if old in old_to_new:
                raise ValueError("induced subgraph vertices must be unique")
            old_to_new[old] = new
        masks = [0] * len(saved_vertices)
        for new, old in enumerate(saved_vertices):
            for old_neighbor in _iter_bits(self._adjacency[old]):
                new_neighbor = old_to_new.get(old_neighbor)
                if new_neighbor is not None:
                    masks[new] |= 1 << new_neighbor
        edge_count = sum(mask.bit_count() for mask in masks) // 2
        return self._from_masks(masks, edge_count)

    def relabel(self, permutation: Sequence[int]) -> Graph:
        """relabel with a mapping from old vertices to new vertices."""
        if len(permutation) != self._vertex_count:
            raise ValueError("a relabeling must contain every vertex")
        if any(not isinstance(v, int) or isinstance(v, bool) for v in permutation):
            raise TypeError("a relabeling must use ordinary integers")
        if set(permutation) != set(range(self._vertex_count)):
            raise ValueError("a relabeling must be a permutation")
        masks = [0] * self._vertex_count
        for old_u, old_v in self.iter_edges():
            new_u = permutation[old_u]
            new_v = permutation[old_v]
            masks[new_u] |= 1 << new_v
            masks[new_v] |= 1 << new_u
        return self._from_masks(masks, self._edge_count)

    def is_isomorphic_to(self, other: Graph) -> bool:
        """check structural equality while ignoring vertex names."""
        from ramsey.tools.isomorphism import are_isomorphic

        return are_isomorphic(self, other)

    def find_isomorphism(self, other: Graph) -> tuple[int, ...] | None:
        """find an old-to-new vertex map into another graph."""
        from ramsey.tools.isomorphism import find_isomorphism

        return find_isomorphism(self, other)

    def canonical_label(self) -> Graph:
        """make a canonical copy that ignores the current vertex names."""
        from ramsey.tools.isomorphism import canonical_label

        return canonical_label(self)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Graph)
            and self._vertex_count == other._vertex_count
            and self._adjacency == other._adjacency
        )

    def __repr__(self) -> str:
        return f"Graph(vertex_count={self._vertex_count}, edge_count={self._edge_count})"
