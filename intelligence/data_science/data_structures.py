"""Core data structures implemented from scratch."""

import heapq
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

__all__ = [
    "Stack", "Queue", "Deque", "LinkedList", "DoublyLinkedList",
    "BST", "AVLTree", "MinHeap", "MaxHeap", "PriorityQueue",
    "DisjointSet", "Graph",
]

T = TypeVar("T")


class Stack(Generic[T]):
    """LIFO stack backed by a Python list."""

    __slots__ = ("_items",)

    def __init__(self) -> None:
        self._items: list = []

    def push(self, item: T) -> None:
        """Push an item onto the stack."""
        self._items.append(item)

    def pop(self) -> T:
        """Remove and return the top item. Raises IndexError if empty."""
        return self._items.pop()

    def peek(self) -> T:
        """Return the top item without removing it. Raises IndexError if empty."""
        return self._items[-1]

    def is_empty(self) -> bool:
        """Return True if the stack is empty."""
        return len(self._items) == 0

    def size(self) -> int:
        """Return the number of items in the stack."""
        return len(self._items)


class Queue(Generic[T]):
    """FIFO queue backed by a Python list."""

    __slots__ = ("_items",)

    def __init__(self) -> None:
        self._items: list = []

    def enqueue(self, item: T) -> None:
        """Add an item to the back of the queue."""
        self._items.append(item)

    def dequeue(self) -> T:
        """Remove and return the front item. Raises IndexError if empty."""
        return self._items.pop(0)

    def peek(self) -> T:
        """Return the front item without removing it. Raises IndexError if empty."""
        return self._items[0]

    def is_empty(self) -> bool:
        """Return True if the queue is empty."""
        return len(self._items) == 0

    def size(self) -> int:
        """Return the number of items in the queue."""
        return len(self._items)


class Deque(Generic[T]):
    """Double-ended queue backed by a Python list."""

    __slots__ = ("_items",)

    def __init__(self) -> None:
        self._items: list = []

    def add_front(self, item: T) -> None:
        """Add an item to the front of the deque."""
        self._items.insert(0, item)

    def add_back(self, item: T) -> None:
        """Add an item to the back of the deque."""
        self._items.append(item)

    def remove_front(self) -> T:
        """Remove and return the front item. Raises IndexError if empty."""
        return self._items.pop(0)

    def remove_back(self) -> T:
        """Remove and return the back item. Raises IndexError if empty."""
        return self._items.pop()

    def size(self) -> int:
        """Return the number of items in the deque."""
        return len(self._items)


class LinkedList(Generic[T]):
    """Singly linked list with generic node type."""

    @dataclass
    class Node(Generic[T]):
        """A node in a singly linked list."""

        data: T
        next: Optional["LinkedList.Node"] = None

    __slots__ = ("head", "_size")

    def __init__(self) -> None:
        self.head: LinkedList.Node | None = None
        self._size: int = 0

    def append(self, data: T) -> None:
        """Append an item to the end of the list."""
        new_node = self.Node(data)
        if not self.head:
            self.head = new_node
        else:
            cur = self.head
            while cur.next:
                cur = cur.next
            cur.next = new_node
        self._size += 1

    def prepend(self, data: T) -> None:
        """Prepend an item to the beginning of the list."""
        self.head = self.Node(data, self.head)
        self._size += 1

    def delete(self, data: T) -> bool:
        """Delete the first occurrence of data. Returns True if found and deleted."""
        if not self.head:
            return False
        if self.head.data == data:
            self.head = self.head.next
            self._size -= 1
            return True
        cur = self.head
        while cur.next:
            if cur.next.data == data:
                cur.next = cur.next.next
                self._size -= 1
                return True
            cur = cur.next
        return False

    def find(self, data: T) -> bool:
        """Return True if data exists in the list."""
        cur = self.head
        while cur:
            if cur.data == data:
                return True
            cur = cur.next
        return False

    def reverse(self) -> None:
        """Reverse the list in-place."""
        prev = None
        cur = self.head
        while cur:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt
        self.head = prev

    def to_list(self) -> list:
        """Convert the linked list to a Python list."""
        result = []
        cur = self.head
        while cur:
            result.append(cur.data)
            cur = cur.next
        return result

    @classmethod
    def from_list(cls, items: list) -> "LinkedList":
        """Create a LinkedList from a Python list."""
        ll = cls()
        for item in items:
            ll.append(item)
        return ll


class DoublyLinkedList(Generic[T]):
    """Doubly linked list with prev/next pointers on each node."""

    @dataclass
    class Node(Generic[T]):
        """A node in a doubly linked list."""

        data: T
        prev: Optional["DoublyLinkedList.Node"] = None
        next: Optional["DoublyLinkedList.Node"] = None

    __slots__ = ("head", "tail", "_size")

    def __init__(self) -> None:
        self.head: DoublyLinkedList.Node | None = None
        self.tail: DoublyLinkedList.Node | None = None
        self._size: int = 0

    def append(self, data: T) -> None:
        """Append an item to the end of the list."""
        new_node = self.Node(data)
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        self._size += 1

    def prepend(self, data: T) -> None:
        """Prepend an item to the beginning of the list."""
        new_node = self.Node(data)
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        self._size += 1

    def delete(self, data: T) -> bool:
        """Delete the first occurrence of data. Returns True if found and deleted."""
        cur = self.head
        while cur:
            if cur.data == data:
                if cur.prev:
                    cur.prev.next = cur.next
                else:
                    self.head = cur.next
                if cur.next:
                    cur.next.prev = cur.prev
                else:
                    self.tail = cur.prev
                self._size -= 1
                return True
            cur = cur.next
        return False

    def to_list(self) -> list:
        """Convert the doubly linked list to a Python list."""
        result = []
        cur = self.head
        while cur:
            result.append(cur.data)
            cur = cur.next
        return result


class BST(Generic[T]):
    """Binary search tree."""

    @dataclass
    class _Node(Generic[T]):
        """A node in a BST."""

        data: T
        left: Optional["BST._Node"] = None
        right: Optional["BST._Node"] = None

    __slots__ = ("root", "_size")

    def __init__(self) -> None:
        self.root: BST._Node | None = None
        self._size: int = 0

    def insert(self, data: T) -> None:
        """Insert a value into the BST."""

        def _insert(node, val):
            if node is None:
                return BST._Node(val)
            if val < node.data:
                node.left = _insert(node.left, val)
            elif val > node.data:
                node.right = _insert(node.right, val)
            return node

        self.root = _insert(self.root, data)
        self._size += 1

    def delete(self, data: T) -> None:
        """Delete a value from the BST if present."""

        def _min_node(node):
            while node.left:
                node = node.left
            return node

        def _delete(node, val):
            if node is None:
                return None
            if val < node.data:
                node.left = _delete(node.left, val)
            elif val > node.data:
                node.right = _delete(node.right, val)
            else:
                if node.left is None:
                    return node.right
                elif node.right is None:
                    return node.left
                successor = _min_node(node.right)
                node.data = successor.data
                node.right = _delete(node.right, successor.data)
            return node

        self.root = _delete(self.root, data)

    def find(self, data: T) -> bool:
        """Return True if data exists in the BST."""
        cur = self.root
        while cur:
            if data == cur.data:
                return True
            elif data < cur.data:
                cur = cur.left
            else:
                cur = cur.right
        return False

    def in_order(self) -> list:
        """Return in-order traversal as a list."""
        result = []

        def _traverse(node):
            if node:
                _traverse(node.left)
                result.append(node.data)
                _traverse(node.right)

        _traverse(self.root)
        return result

    def pre_order(self) -> list:
        """Return pre-order traversal as a list."""
        result = []

        def _traverse(node):
            if node:
                result.append(node.data)
                _traverse(node.left)
                _traverse(node.right)

        _traverse(self.root)
        return result

    def post_order(self) -> list:
        """Return post-order traversal as a list."""
        result = []

        def _traverse(node):
            if node:
                _traverse(node.left)
                _traverse(node.right)
                result.append(node.data)

        _traverse(self.root)
        return result

    def min(self) -> T:
        """Return the minimum value. Raises ValueError if empty."""
        if not self.root:
            raise ValueError("BST is empty")
        cur = self.root
        while cur.left:
            cur = cur.left
        return cur.data

    def max(self) -> T:
        """Return the maximum value. Raises ValueError if empty."""
        if not self.root:
            raise ValueError("BST is empty")
        cur = self.root
        while cur.right:
            cur = cur.right
        return cur.data

    def height(self) -> int:
        """Return the height of the tree (-1 if empty)."""

        def _height(node):
            if node is None:
                return -1
            return 1 + max(_height(node.left), _height(node.right))

        return _height(self.root)

    def size(self) -> int:
        """Return the number of nodes in the tree."""
        return self._size


class AVLTree(Generic[T]):
    """Self-balancing AVL tree."""

    @dataclass
    class _Node(Generic[T]):
        """A node in an AVL tree."""

        data: T
        height: int = 1
        left: Optional["AVLTree._Node"] = None
        right: Optional["AVLTree._Node"] = None

    __slots__ = ("root", "_size")

    def __init__(self) -> None:
        self.root: AVLTree._Node | None = None
        self._size: int = 0

    def _height(self, node: _Node | None) -> int:
        return node.height if node else 0

    def _balance_factor(self, node: _Node) -> int:
        return self._height(node.left) - self._height(node.right)

    def _update_height(self, node: _Node) -> None:
        node.height = 1 + max(self._height(node.left), self._height(node.right))

    def _rotate_right(self, y: _Node) -> _Node:
        x = y.left
        t2 = x.right
        x.right = y
        y.left = t2
        self._update_height(y)
        self._update_height(x)
        return x

    def _rotate_left(self, x: _Node) -> _Node:
        y = x.right
        t2 = y.left
        y.left = x
        x.right = t2
        self._update_height(x)
        self._update_height(y)
        return y

    def _rebalance(self, node: _Node) -> _Node:
        self._update_height(node)
        bf = self._balance_factor(node)
        if bf > 1:
            if self._balance_factor(node.left) < 0:
                node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
        if bf < -1:
            if self._balance_factor(node.right) > 0:
                node.right = self._rotate_right(node.right)
            return self._rotate_left(node)
        return node

    def _insert(self, node: _Node | None, data: T) -> _Node:
        if node is None:
            return self._Node(data)
        if data < node.data:
            node.left = self._insert(node.left, data)
        elif data > node.data:
            node.right = self._insert(node.right, data)
        else:
            return node
        return self._rebalance(node)

    def insert(self, data: T) -> None:
        """Insert a value and rebalance the tree."""
        self.root = self._insert(self.root, data)
        self._size += 1

    def _min_node(self, node: _Node) -> _Node:
        while node.left:
            node = node.left
        return node

    def _delete(self, node: _Node | None, data: T) -> _Node | None:
        if node is None:
            return None
        if data < node.data:
            node.left = self._delete(node.left, data)
        elif data > node.data:
            node.right = self._delete(node.right, data)
        else:
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left
            successor = self._min_node(node.right)
            node.data = successor.data
            node.right = self._delete(node.right, successor.data)
        return self._rebalance(node) if node else None

    def delete(self, data: T) -> None:
        """Delete a value and rebalance the tree."""
        self.root = self._delete(self.root, data)

    def find(self, data: T) -> bool:
        """Return True if data exists in the AVL tree."""
        cur = self.root
        while cur:
            if data == cur.data:
                return True
            elif data < cur.data:
                cur = cur.left
            else:
                cur = cur.right
        return False

    def height(self) -> int:
        """Return the height of the AVL tree (-1 if empty)."""
        return self._height(self.root) - 1 if self.root else -1

    def in_order(self) -> list:
        """Return in-order traversal as a list."""
        result = []

        def _traverse(node):
            if node:
                _traverse(node.left)
                result.append(node.data)
                _traverse(node.right)

        _traverse(self.root)
        return result


class MinHeap(Generic[T]):
    """Binary min-heap using a list."""

    __slots__ = ("_data",)

    def __init__(self) -> None:
        self._data: list = []

    def push(self, item: T) -> None:
        """Push an item onto the heap."""
        self._data.append(item)
        self._sift_up(len(self._data) - 1)

    def pop(self) -> T:
        """Remove and return the minimum item. Raises IndexError if empty."""
        if len(self._data) == 1:
            return self._data.pop()
        top = self._data[0]
        self._data[0] = self._data.pop()
        self._sift_down(0)
        return top

    def peek(self) -> T:
        """Return the minimum item without removing it."""
        return self._data[0]

    def size(self) -> int:
        """Return the number of items in the heap."""
        return len(self._data)

    def _sift_up(self, i: int) -> None:
        while i > 0:
            parent = (i - 1) // 2
            if self._data[i] < self._data[parent]:
                self._data[i], self._data[parent] = self._data[parent], self._data[i]
                i = parent
            else:
                break

    def _sift_down(self, i: int) -> None:
        n = len(self._data)
        while True:
            smallest = i
            left = 2 * i + 1
            right = 2 * i + 2
            if left < n and self._data[left] < self._data[smallest]:
                smallest = left
            if right < n and self._data[right] < self._data[smallest]:
                smallest = right
            if smallest == i:
                break
            self._data[i], self._data[smallest] = self._data[smallest], self._data[i]
            i = smallest


def heapify_min(items: list) -> MinHeap:
    """Build a MinHeap from a list of items in O(n)."""
    h = MinHeap()
    h._data = list(items)
    for i in range(len(h._data) // 2 - 1, -1, -1):
        h._sift_down(i)
    return h


class MaxHeap(Generic[T]):
    """Binary max-heap using a list."""

    __slots__ = ("_data",)

    def __init__(self) -> None:
        self._data: list = []

    def push(self, item: T) -> None:
        """Push an item onto the heap."""
        self._data.append(item)
        self._sift_up(len(self._data) - 1)

    def pop(self) -> T:
        """Remove and return the maximum item. Raises IndexError if empty."""
        if len(self._data) == 1:
            return self._data.pop()
        top = self._data[0]
        self._data[0] = self._data.pop()
        self._sift_down(0)
        return top

    def peek(self) -> T:
        """Return the maximum item without removing it."""
        return self._data[0]

    def size(self) -> int:
        """Return the number of items in the heap."""
        return len(self._data)

    def _sift_up(self, i: int) -> None:
        while i > 0:
            parent = (i - 1) // 2
            if self._data[i] > self._data[parent]:
                self._data[i], self._data[parent] = self._data[parent], self._data[i]
                i = parent
            else:
                break

    def _sift_down(self, i: int) -> None:
        n = len(self._data)
        while True:
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2
            if left < n and self._data[left] > self._data[largest]:
                largest = left
            if right < n and self._data[right] > self._data[largest]:
                largest = right
            if largest == i:
                break
            self._data[i], self._data[largest] = self._data[largest], self._data[i]
            i = largest


@dataclass
class _PQItem(Generic[T]):
    """Internal item wrapper for priority queue ordering."""

    priority: float
    item: T

    def __lt__(self, other: "_PQItem") -> bool:
        return self.priority < other.priority


class PriorityQueue(Generic[T]):
    """Priority queue backed by a min-heap."""

    __slots__ = ("_heap",)

    def __init__(self) -> None:
        self._heap: list = []

    def push(self, item: T, priority: float) -> None:
        """Add an item with the given priority (lower = higher priority)."""
        entry = _PQItem(priority, item)
        heapq.heappush(self._heap, entry)

    def pop(self) -> T:
        """Remove and return the highest-priority (lowest priority value) item."""
        return heapq.heappop(self._heap).item

    def peek(self) -> T:
        """Return the highest-priority item without removing it."""
        return self._heap[0].item

    def size(self) -> int:
        """Return the number of items in the priority queue."""
        return len(self._heap)


class DisjointSet(Generic[T]):
    """Union-Find with path compression and union by rank."""

    __slots__ = ("_parent", "_rank", "_components")

    def __init__(self) -> None:
        self._parent: dict = {}
        self._rank: dict = {}
        self._components: int = 0

    def _make_set(self, x: T) -> None:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
            self._components += 1

    def find(self, x: T) -> T:
        """Find the representative of the set containing x."""
        self._make_set(x)
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, x: T, y: T) -> None:
        """Merge the sets containing x and y."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1
        self._components -= 1

    def connected(self, x: T, y: T) -> bool:
        """Return True if x and y are in the same set."""
        return self.find(x) == self.find(y)

    def components(self) -> int:
        """Return the number of disjoint sets."""
        return self._components


class Graph:
    """Undirected graph using an adjacency list representation."""

    __slots__ = ("adjacency",)

    def __init__(self) -> None:
        self.adjacency: dict = {}

    def add_vertex(self, v) -> None:
        """Add a vertex to the graph."""
        if v not in self.adjacency:
            self.adjacency[v] = []

    def add_edge(self, u, v) -> None:
        """Add an undirected edge between u and v."""
        self.add_vertex(u)
        self.add_vertex(v)
        if v not in self.adjacency[u]:
            self.adjacency[u].append(v)
        if u not in self.adjacency[v]:
            self.adjacency[v].append(u)

    def bfs(self, start) -> list:
        """Breadth-first search traversal returning visited vertices in order."""
        visited = set()
        queue = [start]
        visited.add(start)
        result = []
        while queue:
            vertex = queue.pop(0)
            result.append(vertex)
            for neighbor in self.adjacency.get(vertex, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return result

    def dfs(self, start) -> list:
        """Depth-first search traversal returning visited vertices in order."""
        visited = set()
        result = []

        def _dfs(v):
            visited.add(v)
            result.append(v)
            for neighbor in self.adjacency.get(v, []):
                if neighbor not in visited:
                    _dfs(neighbor)

        _dfs(start)
        return result

    def has_path(self, u, v) -> bool:
        """Return True if there is a path from u to v."""
        if u not in self.adjacency or v not in self.adjacency:
            return False
        visited = set()
        stack = [u]
        while stack:
            node = stack.pop()
            if node == v:
                return True
            if node not in visited:
                visited.add(node)
                for neighbor in self.adjacency.get(node, []):
                    stack.append(neighbor)
        return False

    def shortest_path_bfs(self, start, end) -> list:
        """Return the shortest path from start to end using BFS, or [] if none."""
        if start not in self.adjacency or end not in self.adjacency:
            return []
        if start == end:
            return [start]
        visited = {start}
        queue = [[start]]
        while queue:
            path = queue.pop(0)
            node = path[-1]
            for neighbor in self.adjacency.get(node, []):
                if neighbor == end:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
        return []

    def topological_sort(self) -> list:
        """Topological sort using DFS (only valid for DAGs).

        Raises ValueError if the graph contains a cycle.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color = dict.fromkeys(self.adjacency, WHITE)
        order = []

        def _dfs(v):
            color[v] = GRAY
            for neighbor in self.adjacency.get(v, []):
                if color[neighbor] == GRAY:
                    raise ValueError("Graph contains a cycle")
                if color[neighbor] == WHITE:
                    _dfs(neighbor)
            color[v] = BLACK
            order.append(v)

        for v in self.adjacency:
            if color[v] == WHITE:
                _dfs(v)
        return list(reversed(order))

    def has_cycle(self) -> bool:
        """Return True if the undirected graph contains a cycle."""
        visited = set()

        def _has_cycle(v, parent):
            visited.add(v)
            for neighbor in self.adjacency.get(v, []):
                if neighbor not in visited:
                    if _has_cycle(neighbor, v):
                        return True
                elif neighbor != parent:
                    return True
            return False

        for vertex in self.adjacency:
            if vertex not in visited:
                if _has_cycle(vertex, None):
                    return True
        return False

    def connected_components(self) -> list:
        """Return a list of lists, each containing vertices of a connected component."""
        visited = set()
        components = []

        def _dfs(v, component):
            visited.add(v)
            component.append(v)
            for neighbor in self.adjacency.get(v, []):
                if neighbor not in visited:
                    _dfs(neighbor, component)

        for vertex in self.adjacency:
            if vertex not in visited:
                component = []
                _dfs(vertex, component)
                components.append(component)
        return components

    def mst_kruskal(self) -> list:
        """Find minimum spanning tree using Kruskal's algorithm.

        Returns a list of (u, v, weight) tuples. Edge weights are stored
        as tuples (neighbor, weight) in the adjacency list; plain neighbor
        entries are treated as weight 1.
        """
        edges = []
        for u in self.adjacency:
            for entry in self.adjacency[u]:
                if isinstance(entry, tuple):
                    v, w = entry
                else:
                    v, w = entry, 1
                if (u, v) not in [(a, b) for a, b, _ in edges] and \
                   (v, u) not in [(a, b) for a, b, _ in edges]:
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
