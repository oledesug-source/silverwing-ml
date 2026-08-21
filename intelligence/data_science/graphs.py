"""Advanced graph algorithms for weighted graphs and network flows."""

import heapq
from collections import defaultdict
from collections.abc import Callable

__all__ = [
    "GraphWeighted", "ford_fulkerson", "min_cost_max_flow",
    "bipartite_check", "strongly_connected_components", "a_star",
]


class GraphWeighted:
    """Weighted directed graph using adjacency lists.

    Each vertex maps to a list of ``(neighbor, weight)`` tuples.
    """

    __slots__ = ("adjacency",)

    def __init__(self) -> None:
        self.adjacency: dict = {}

    def add_vertex(self, v) -> None:
        """Add a vertex to the graph."""
        if v not in self.adjacency:
            self.adjacency[v] = []

    def add_edge(self, u, v, weight: float) -> None:
        """Add a directed edge from u to v with the given weight."""
        self.add_vertex(u)
        self.add_vertex(v)
        self.adjacency[u].append((v, weight))

    def dijkstra(self, source) -> dict:
        """Dijkstra's single-source shortest-path algorithm.

        Returns a dict mapping each vertex to its shortest distance from source.
        """
        dist = {v: float("inf") for v in self.adjacency}
        dist[source] = 0
        pq = [(0, source)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, w in self.adjacency.get(u, []):
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    heapq.heappush(pq, (dist[v], v))
        return dist

    def bellman_ford(self, source) -> dict:
        """Bellman-Ford single-source shortest-path algorithm.

        Returns a dict of distances. Raises ValueError if a negative cycle exists.
        """
        dist = {v: float("inf") for v in self.adjacency}
        dist[source] = 0

        edges = []
        for u in self.adjacency:
            for v, w in self.adjacency[u]:
                edges.append((u, v, w))

        for _ in range(len(self.adjacency) - 1):
            for u, v, w in edges:
                if dist[u] != float("inf") and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w

        for u, v, w in edges:
            if dist[u] != float("inf") and dist[u] + w < dist[v]:
                raise ValueError("Graph contains a negative weight cycle")

        return dist

    def floyd_warshall(self) -> dict:
        """Floyd-Warshall all-pairs shortest-path algorithm.

        Returns a nested dict ``dist[u][v]`` with shortest distances.
        """
        vertices = list(self.adjacency.keys())
        dist = {u: {v: float("inf") for v in vertices} for u in vertices}

        for v in vertices:
            dist[v][v] = 0
        for u in self.adjacency:
            for v, w in self.adjacency[u]:
                dist[u][v] = w

        for k in vertices:
            for i in vertices:
                for j in vertices:
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        return dist

    def prim_mst(self) -> list:
        """Prim's minimum spanning tree algorithm.

        Returns a list of ``(u, v, weight)`` edges.
        """
        if not self.adjacency:
            return []

        start = next(iter(self.adjacency))
        visited = {start}
        edges = [(w, start, v) for v, w in self.adjacency.get(start, [])]
        heapq.heapify(edges)
        mst = []

        while edges:
            w, u, v = heapq.heappop(edges)
            if v in visited:
                continue
            visited.add(v)
            mst.append((u, v, w))
            for neighbor, weight in self.adjacency.get(v, []):
                if neighbor not in visited:
                    heapq.heappush(edges, (weight, v, neighbor))
        return mst

    def kruskal_mst(self) -> list:
        """Kruskal's minimum spanning tree algorithm.

        Returns a list of ``(u, v, weight)`` edges.
        """
        from .data_structures import DisjointSet

        edges = []
        for u in self.adjacency:
            for v, w in self.adjacency[u]:
                if (v, u, w) not in edges:
                    edges.append((u, v, w))

        edges.sort(key=lambda e: e[2])
        ds = DisjointSet()
        for v in self.adjacency:
            ds.find(v)

        mst = []
        for u, v, w in edges:
            if not ds.connected(u, v):
                ds.union(u, v)
                mst.append((u, v, w))
        return mst


def ford_fulkerson(adjacency: dict, source, sink) -> int:
    """Edmonds-Karp BFS-based maximum flow algorithm.

    ``adjacency`` maps each vertex to a dict of ``{neighbor: capacity}``.
    Returns the maximum flow value from source to sink.
    """
    residual = {u: dict(neighbors) for u, neighbors in adjacency.items()}
    for u in list(residual.keys()):
        for v in list(residual[u].keys()):
            if v not in residual:
                residual[v] = {}
            if v not in residual[u]:
                residual[u][v] = 0

    max_flow = 0

    while True:
        parent = {source: None}
        queue = [source]
        found = False
        while queue:
            u = queue.pop(0)
            for v, cap in residual.get(u, {}).items():
                if v not in parent and cap > 0:
                    parent[v] = u
                    if v == sink:
                        found = True
                        break
                    queue.append(v)
            if found:
                break
        if not found:
            break

        path_flow = float("inf")
        v = sink
        while v != source:
            u = parent[v]
            path_flow = min(path_flow, residual[u][v])
            v = u

        v = sink
        while v != source:
            u = parent[v]
            residual[u][v] -= path_flow
            residual[v][u] = residual.get(v, {}).get(u, 0) + path_flow
            v = u

        max_flow += path_flow

    return max_flow


def min_cost_max_flow(
    num_nodes: int,
    edges: list,
    source: int,
    sink: int,
) -> tuple[int, int]:
    """Simplified minimum-cost maximum-flow using successive shortest paths.

    ``edges`` is a list of ``(u, v, capacity, cost)`` tuples.
    Returns ``(max_flow, min_cost)``.
    """
    adj = defaultdict(list)
    cap = {}
    cost = {}
    for u, v, c, w in edges:
        idx_f = len(adj[u])
        idx_b = len(adj[v])
        adj[u].append((v, idx_b))
        adj[v].append((u, idx_f))
        cap[(u, idx_f)] = c
        cost[(u, idx_f)] = w
        cap[(v, idx_b)] = 0
        cost[(v, idx_b)] = -w

    total_flow = 0
    total_cost = 0

    while True:
        dist = [float("inf")] * num_nodes
        prev_v = [-1] * num_nodes
        prev_e = [-1] * num_nodes
        dist[source] = 0
        in_queue = [True] * num_nodes
        queue = [source]

        while queue:
            u = queue.pop(0)
            in_queue[u] = False
            for i, (v, _) in enumerate(adj[u]):
                if cap[(u, i)] > 0 and dist[u] + cost[(u, i)] < dist[v]:
                    dist[v] = dist[u] + cost[(u, i)]
                    prev_v[v] = u
                    prev_e[v] = i
                    if not in_queue[v]:
                        in_queue[v] = True
                        queue.append(v)

        if dist[sink] == float("inf"):
            break

        path_flow = float("inf")
        v = sink
        while v != source:
            u = prev_v[v]
            e = prev_e[v]
            path_flow = min(path_flow, cap[(u, e)])
            v = u

        v = sink
        while v != source:
            u = prev_v[v]
            e = prev_e[v]
            cap[(u, e)] -= path_flow
            rev_e = prev_e[u] if prev_v[u] != -1 else 0
            cap[(v, rev_e)] += path_flow if (v, rev_e) in cap else path_flow
            total_cost += cost[(u, e)] * path_flow
            v = u

        total_flow += path_flow

    return total_flow, total_cost


def bipartite_check(adjacency: dict) -> bool:
    """Check whether an undirected graph is bipartite using BFS 2-coloring."""
    color = {}
    for start in adjacency:
        if start in color:
            continue
        color[start] = 0
        queue = [start]
        while queue:
            u = queue.pop(0)
            for v in adjacency.get(u, []):
                if v not in color:
                    color[v] = 1 - color[u]
                    queue.append(v)
                elif color[v] == color[u]:
                    return False
    return True


def strongly_connected_components(adjacency: dict) -> list:
    """Tarjan's algorithm for finding strongly connected components.

    Returns a list of lists, each containing vertices of one SCC.
    """
    index_counter = [0]
    stack = []
    lowlink = {}
    index = {}
    on_stack = {}
    result = []

    def strongconnect(v):
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True

        for w in adjacency.get(v, []):
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack.get(w, False):
                lowlink[v] = min(lowlink[v], index[w])

        if lowlink[v] == index[v]:
            component = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                component.append(w)
                if w == v:
                    break
            result.append(component)

    for v in adjacency:
        if v not in index:
            strongconnect(v)

    return result


def a_star(
    grid: list,
    start: tuple,
    goal: tuple,
    heuristic: Callable | None = None,
) -> list:
    """A* search on a 2-D grid.

    ``grid`` is a list of lists where 0 is walkable and 1 is blocked.
    ``start`` and ``goal`` are ``(row, col)`` tuples.
    ``heuristic`` defaults to Manhattan distance.
    Returns the path as a list of ``(row, col)`` tuples, or [] if unreachable.
    """
    if heuristic is None:
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    open_set = [(heuristic(start, goal), 0, start)]
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, g, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return list(reversed(path))

        for dr, dc in directions:
            nr, nc = current[0] + dr, current[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
                neighbor = (nr, nc)
                tentative_g = g + 1
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, tentative_g, neighbor))

    return []
