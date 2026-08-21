"""Comprehensive tests for the intelligence.data_science module."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from intelligence.data_science.complexity import (
    amortized_analysis,
    measure_complexity,
    space_complexity,
)
from intelligence.data_science.data_structures import (
    BST,
    AVLTree,
    Deque,
    DisjointSet,
    DoublyLinkedList,
    Graph,
    LinkedList,
    MaxHeap,
    MinHeap,
    PriorityQueue,
    Queue,
    Stack,
    heapify_min,
)
from intelligence.data_science.dynamic_programming import (
    coin_change,
    edit_distance,
    knapsack,
    lcs,
    lis,
    lis_nlogn,
    longest_palindrome,
    matrix_chain_order,
    rod_cutting,
    subset_sum,
)
from intelligence.data_science.dynamic_programming import (
    fibonacci as fib,
)
from intelligence.data_science.graphs import (
    GraphWeighted,
    a_star,
    bipartite_check,
    ford_fulkerson,
    strongly_connected_components,
)
from intelligence.data_science.searching import (
    Trie,
    binary_search,
    binary_search_left,
    binary_search_right,
    boyer_moore_search,
    exponential_search,
    fibonacci_search,
    interpolation_search,
    jump_search,
    kmp_search,
)
from intelligence.data_science.sorting import (
    SortResult,
    bucket_sort,
    counting_sort,
    get_sort_stats,
    heapsort,
    mergesort,
    quicksort,
    radix_sort,
    tim_sort,
    tim_sort_run,
)


class TestSorting(unittest.TestCase):

    def _check_sorted(self, arr):
        return arr == sorted(arr)

    def test_quicksort_empty(self):
        self.assertEqual(quicksort([]), [])

    def test_quicksort_single(self):
        self.assertEqual(quicksort([1]), [1])

    def test_quicksort_sorted(self):
        self.assertEqual(quicksort([1, 2, 3, 4, 5]), [1, 2, 3, 4, 5])

    def test_quicksort_reverse(self):
        self.assertEqual(quicksort([5, 4, 3, 2, 1]), [1, 2, 3, 4, 5])

    def test_quicksort_random(self):
        data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
        self.assertEqual(quicksort(data), sorted(data))

    def test_quicksort_duplicates(self):
        self.assertEqual(quicksort([2, 2, 2, 1, 1]), [1, 1, 2, 2, 2])

    def test_mergesort_empty(self):
        self.assertEqual(mergesort([]), [])

    def test_mergesort_single(self):
        self.assertEqual(mergesort([42]), [42])

    def test_mergesort_sorted(self):
        self.assertEqual(mergesort([1, 2, 3, 4]), [1, 2, 3, 4])

    def test_mergesort_reverse(self):
        self.assertEqual(mergesort([4, 3, 2, 1]), [1, 2, 3, 4])

    def test_mergesort_random(self):
        data = [7, 3, 5, 1, 9, 2, 8, 4, 6]
        self.assertEqual(mergesort(data), sorted(data))

    def test_heapsort_empty(self):
        self.assertEqual(heapsort([]), [])

    def test_heapsort_single(self):
        self.assertEqual(heapsort([1]), [1])

    def test_heapsort_sorted(self):
        self.assertEqual(heapsort([1, 2, 3, 4, 5]), [1, 2, 3, 4, 5])

    def test_heapsort_reverse(self):
        self.assertEqual(heapsort([5, 4, 3, 2, 1]), [1, 2, 3, 4, 5])

    def test_heapsort_random(self):
        data = [9, 3, 7, 1, 5, 2, 8, 4, 6]
        self.assertEqual(heapsort(data), sorted(data))

    def test_tim_sort_empty(self):
        self.assertEqual(tim_sort([]), [])

    def test_tim_sort_single(self):
        self.assertEqual(tim_sort([1]), [1])

    def test_tim_sort_sorted(self):
        self.assertEqual(tim_sort([1, 2, 3, 4, 5]), [1, 2, 3, 4, 5])

    def test_tim_sort_reverse(self):
        self.assertEqual(tim_sort([5, 4, 3, 2, 1]), [1, 2, 3, 4, 5])

    def test_tim_sort_random(self):
        data = [3, 1, 4, 1, 5, 9, 2, 6]
        self.assertEqual(tim_sort(data), sorted(data))

    def test_tim_sort_run_detection(self):
        arr = [1, 2, 3, 8, 7, 6, 5]
        run_len = tim_sort_run(arr, 0, 7)
        self.assertEqual(run_len, 4)

    def test_counting_sort(self):
        self.assertEqual(counting_sort([4, 2, 2, 8, 3, 3, 1], 8), [1, 2, 2, 3, 3, 4, 8])

    def test_counting_sort_empty(self):
        self.assertEqual(counting_sort([], 5), [])

    def test_radix_sort(self):
        self.assertEqual(radix_sort([170, 45, 75, 90, 802, 24, 2, 66]), [2, 24, 45, 66, 75, 90, 170, 802])

    def test_radix_sort_empty(self):
        self.assertEqual(radix_sort([]), [])

    def test_bucket_sort(self):
        data = [0.42, 0.32, 0.23, 0.52, 0.25, 0.47, 0.51]
        result = bucket_sort(data)
        self.assertTrue(self._check_sorted(result))

    def test_bucket_sort_empty(self):
        self.assertEqual(bucket_sort([]), [])

    def test_sort_result_dataclass(self):
        sr = SortResult(sorted_list=[1, 2], comparisons=5, swaps=3)
        self.assertEqual(sr.sorted_list, [1, 2])
        self.assertEqual(sr.comparisons, 5)
        self.assertEqual(sr.swaps, 3)

    def test_get_sort_stats(self):
        stats = get_sort_stats()
        self.assertIn("quicksort", stats)
        self.assertIn("mergesort", stats)
        self.assertIn("heapsort", stats)
        self.assertIn("tim_sort", stats)
        self.assertIn("bucket_sort", stats)
        for _name, result in stats.items():
            self.assertIsInstance(result, SortResult)
            self.assertTrue(self._check_sorted(result.sorted_list))


class TestSearching(unittest.TestCase):

    def test_binary_search_found(self):
        arr = [1, 2, 3, 4, 5]
        self.assertEqual(binary_search(arr, 3), 2)

    def test_binary_search_not_found(self):
        arr = [1, 2, 3, 4, 5]
        self.assertEqual(binary_search(arr, 6), -1)

    def test_binary_search_left(self):
        arr = [1, 2, 2, 2, 3]
        self.assertEqual(binary_search_left(arr, 2), 1)

    def test_binary_search_left_beyond(self):
        arr = [1, 2, 2, 2, 3]
        self.assertEqual(binary_search_left(arr, 4), 5)

    def test_binary_search_right(self):
        arr = [1, 2, 2, 2, 3]
        self.assertEqual(binary_search_right(arr, 2), 4)

    def test_interpolation_search_found(self):
        arr = [10, 20, 30, 40, 50]
        self.assertEqual(interpolation_search(arr, 30), 2)

    def test_interpolation_search_not_found(self):
        arr = [10, 20, 30, 40, 50]
        self.assertEqual(interpolation_search(arr, 25), -1)

    def test_exponential_search_found(self):
        arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertEqual(exponential_search(arr, 7), 6)

    def test_exponential_search_not_found(self):
        arr = [1, 2, 3, 4, 5]
        self.assertEqual(exponential_search(arr, 11), -1)

    def test_jump_search_found(self):
        arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertEqual(jump_search(arr, 7), 6)

    def test_jump_search_not_found(self):
        arr = [1, 2, 3, 4, 5]
        self.assertEqual(jump_search(arr, 11), -1)

    def test_fibonacci_search_found(self):
        arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertEqual(fibonacci_search(arr, 5), 4)

    def test_fibonacci_search_not_found(self):
        arr = [1, 2, 3, 4, 5]
        self.assertEqual(fibonacci_search(arr, 11), -1)

    def test_kmp_search(self):
        text = "ABABDABACDABABCABAB"
        pattern = "ABABCABAB"
        self.assertEqual(kmp_search(text, pattern), [10])

    def test_kmp_search_multiple(self):
        text = "AAAA"
        pattern = "AA"
        self.assertEqual(kmp_search(text, pattern), [0, 1, 2])

    def test_kmp_search_not_found(self):
        self.assertEqual(kmp_search("ABCD", "XYZ"), [])

    def test_boyer_moore_search(self):
        text = "ABABDABACDABABCABAB"
        pattern = "ABABCABAB"
        self.assertEqual(boyer_moore_search(text, pattern), [10])

    def test_boyer_moore_search_multiple(self):
        text = "AAAA"
        pattern = "AA"
        result = boyer_moore_search(text, pattern)
        self.assertIn(0, result)

    def test_trie_insert_search(self):
        t = Trie()
        t.insert("hello")
        t.insert("help")
        self.assertTrue(t.search("hello"))
        self.assertTrue(t.search("help"))
        self.assertFalse(t.search("hell"))

    def test_trie_starts_with(self):
        t = Trie()
        t.insert("hello")
        self.assertTrue(t.starts_with("hel"))
        self.assertFalse(t.starts_with("xyz"))

    def test_trie_delete(self):
        t = Trie()
        t.insert("hello")
        t.insert("help")
        t.delete("hello")
        self.assertFalse(t.search("hello"))
        self.assertTrue(t.search("help"))

    def test_trie_autocomplete(self):
        t = Trie()
        t.insert("car")
        t.insert("card")
        t.insert("care")
        t.insert("cat")
        result = t.autocomplete("car")
        self.assertEqual(sorted(result), ["car", "card", "care"])

    def test_trie_autocomplete_empty(self):
        t = Trie()
        self.assertEqual(t.autocomplete("xyz"), [])


class TestDataStructures(unittest.TestCase):

    def test_stack(self):
        s = Stack()
        s.push(1)
        s.push(2)
        s.push(3)
        self.assertEqual(s.peek(), 3)
        self.assertEqual(s.pop(), 3)
        self.assertEqual(s.size(), 2)
        self.assertFalse(s.is_empty())

    def test_stack_empty_pop(self):
        s = Stack()
        with self.assertRaises(IndexError):
            s.pop()

    def test_queue(self):
        q = Queue()
        q.enqueue("a")
        q.enqueue("b")
        q.enqueue("c")
        self.assertEqual(q.peek(), "a")
        self.assertEqual(q.dequeue(), "a")
        self.assertEqual(q.size(), 2)
        self.assertFalse(q.is_empty())

    def test_deque(self):
        d = Deque()
        d.add_back(1)
        d.add_back(2)
        d.add_front(0)
        self.assertEqual(d.size(), 3)
        self.assertEqual(d.remove_front(), 0)
        self.assertEqual(d.remove_back(), 2)
        self.assertEqual(d.size(), 1)

    def test_linked_list(self):
        ll = LinkedList.from_list([1, 2, 3])
        self.assertEqual(ll.to_list(), [1, 2, 3])
        ll.prepend(0)
        self.assertEqual(ll.to_list(), [0, 1, 2, 3])
        ll.delete(2)
        self.assertEqual(ll.to_list(), [0, 1, 3])
        self.assertTrue(ll.find(3))
        self.assertFalse(ll.find(5))

    def test_linked_list_reverse(self):
        ll = LinkedList.from_list([1, 2, 3])
        ll.reverse()
        self.assertEqual(ll.to_list(), [3, 2, 1])

    def test_doubly_linked_list(self):
        dll = DoublyLinkedList()
        dll.append(1)
        dll.append(2)
        dll.prepend(0)
        self.assertEqual(dll.to_list(), [0, 1, 2])
        dll.delete(1)
        self.assertEqual(dll.to_list(), [0, 2])

    def test_bst(self):
        bst = BST()
        for v in [5, 3, 7, 1, 4, 6, 8]:
            bst.insert(v)
        self.assertEqual(bst.in_order(), [1, 3, 4, 5, 6, 7, 8])
        self.assertEqual(bst.pre_order(), [5, 3, 1, 4, 7, 6, 8])
        self.assertEqual(bst.post_order(), [1, 4, 3, 6, 8, 7, 5])
        self.assertTrue(bst.find(4))
        self.assertFalse(bst.find(9))
        self.assertEqual(bst.min(), 1)
        self.assertEqual(bst.max(), 8)
        self.assertEqual(bst.height(), 2)
        self.assertEqual(bst.size(), 7)

    def test_bst_delete(self):
        bst = BST()
        for v in [5, 3, 7, 1, 4]:
            bst.insert(v)
        bst.delete(3)
        self.assertFalse(bst.find(3))
        self.assertTrue(bst.find(5))

    def test_avl_tree(self):
        avl = AVLTree()
        for v in [10, 20, 30, 40, 50, 25]:
            avl.insert(v)
        self.assertEqual(avl.in_order(), [10, 20, 25, 30, 40, 50])
        self.assertTrue(avl.find(25))
        self.assertFalse(avl.find(99))

    def test_avl_tree_delete(self):
        avl = AVLTree()
        for v in [30, 20, 40, 10, 25, 35, 50]:
            avl.insert(v)
        avl.delete(20)
        self.assertFalse(avl.find(20))
        self.assertEqual(avl.in_order(), [10, 25, 30, 35, 40, 50])

    def test_min_heap(self):
        h = MinHeap()
        for v in [5, 3, 7, 1, 4]:
            h.push(v)
        self.assertEqual(h.peek(), 1)
        self.assertEqual(h.pop(), 1)
        self.assertEqual(h.pop(), 3)
        self.assertEqual(h.size(), 3)

    def test_max_heap(self):
        h = MaxHeap()
        for v in [5, 3, 7, 1, 4]:
            h.push(v)
        self.assertEqual(h.peek(), 7)
        self.assertEqual(h.pop(), 7)
        self.assertEqual(h.pop(), 5)

    def test_heapify_min(self):
        h = heapify_min([5, 3, 7, 1, 4])
        result = []
        while h.size() > 0:
            result.append(h.pop())
        self.assertEqual(result, [1, 3, 4, 5, 7])

    def test_priority_queue(self):
        pq = PriorityQueue()
        pq.push("low", 10)
        pq.push("high", 1)
        pq.push("medium", 5)
        self.assertEqual(pq.pop(), "high")
        self.assertEqual(pq.pop(), "medium")
        self.assertEqual(pq.pop(), "low")

    def test_disjoint_set(self):
        ds = DisjointSet()
        ds.union("A", "B")
        ds.union("C", "D")
        self.assertTrue(ds.connected("A", "B"))
        self.assertFalse(ds.connected("A", "C"))
        self.assertEqual(ds.components(), 2)
        ds.union("B", "C")
        self.assertTrue(ds.connected("A", "D"))
        self.assertEqual(ds.components(), 1)

    def test_graph_bfs_dfs(self):
        g = Graph()
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("B", "D")
        g.add_edge("C", "D")
        bfs_result = g.bfs("A")
        self.assertIn("A", bfs_result)
        self.assertIn("B", bfs_result)
        self.assertIn("C", bfs_result)
        self.assertIn("D", bfs_result)
        dfs_result = g.dfs("A")
        self.assertIn("A", dfs_result)
        self.assertIn("D", dfs_result)

    def test_graph_has_path(self):
        g = Graph()
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        self.assertTrue(g.has_path("A", "C"))
        self.assertFalse(g.has_path("A", "Z"))

    def test_graph_shortest_path(self):
        g = Graph()
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("A", "C")
        path = g.shortest_path_bfs("A", "C")
        self.assertIn("A", path)
        self.assertIn("C", path)
        self.assertTrue(len(path) <= 3)

    def test_graph_topological_sort(self):
        g = Graph()
        g.adjacency = {
            "A": ["B", "C"],
            "B": ["D"],
            "C": ["D"],
            "D": [],
        }
        order = g.topological_sort()
        self.assertTrue(order.index("A") < order.index("B"))
        self.assertTrue(order.index("A") < order.index("C"))
        self.assertTrue(order.index("B") < order.index("D"))

    def test_graph_has_cycle(self):
        g_no = Graph()
        g_no.add_edge("A", "B")
        g_no.add_edge("B", "C")
        self.assertFalse(g_no.has_cycle())

        g_yes = Graph()
        g_yes.add_edge("A", "B")
        g_yes.add_edge("B", "C")
        g_yes.add_edge("C", "A")
        self.assertTrue(g_yes.has_cycle())

    def test_graph_connected_components(self):
        g = Graph()
        g.add_edge("A", "B")
        g.add_edge("C", "D")
        g.add_vertex("E")
        components = g.connected_components()
        self.assertEqual(len(components), 3)

    def test_graph_kruskal_mst(self):
        g = Graph()
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("A", "C")
        mst = g.mst_kruskal()
        self.assertEqual(len(mst), 2)


class TestWeightedGraphs(unittest.TestCase):

    def _make_graph(self):
        gw = GraphWeighted()
        gw.add_edge("A", "B", 4)
        gw.add_edge("A", "C", 1)
        gw.add_edge("C", "B", 2)
        gw.add_edge("B", "D", 1)
        gw.add_edge("C", "D", 5)
        return gw

    def test_dijkstra(self):
        gw = self._make_graph()
        dist = gw.dijkstra("A")
        self.assertEqual(dist["A"], 0)
        self.assertEqual(dist["C"], 1)
        self.assertEqual(dist["B"], 3)
        self.assertEqual(dist["D"], 4)

    def test_bellman_ford(self):
        gw = self._make_graph()
        dist = gw.bellman_ford("A")
        self.assertEqual(dist["A"], 0)
        self.assertEqual(dist["C"], 1)
        self.assertEqual(dist["D"], 4)

    def test_bellman_ford_negative_cycle(self):
        gw = GraphWeighted()
        gw.add_edge("A", "B", -1)
        gw.add_edge("B", "C", -1)
        gw.add_edge("C", "A", -1)
        with self.assertRaises(ValueError):
            gw.bellman_ford("A")

    def test_floyd_warshall(self):
        gw = self._make_graph()
        dist = gw.floyd_warshall()
        self.assertEqual(dist["A"]["A"], 0)
        self.assertEqual(dist["A"]["C"], 1)
        self.assertEqual(dist["A"]["D"], 4)

    def test_prim_mst(self):
        gw = self._make_graph()
        mst = gw.prim_mst()
        self.assertEqual(len(mst), 3)

    def test_kruskal_mst(self):
        gw = self._make_graph()
        mst = gw.kruskal_mst()
        self.assertEqual(len(mst), 3)

    def test_ford_fulkerson(self):
        capacity = {
            "s": {"a": 10, "b": 8},
            "a": {"b": 5, "t": 7},
            "b": {"t": 10},
            "t": {},
        }
        flow = ford_fulkerson(capacity, "s", "t")
        self.assertEqual(flow, 17)

    def test_bipartite_check(self):
        g = {
            "A": ["B", "C"],
            "B": ["A", "D"],
            "C": ["A", "D"],
            "D": ["B", "C"],
        }
        self.assertTrue(bipartite_check(g))

        g2 = {
            "A": ["B", "C"],
            "B": ["A", "C"],
            "C": ["A", "B"],
        }
        self.assertFalse(bipartite_check(g2))

    def test_scc(self):
        g = {
            "0": ["1"],
            "1": ["2"],
            "2": ["0", "3"],
            "3": ["4"],
            "4": [],
        }
        sccs = strongly_connected_components(g)
        self.assertEqual(len(sccs), 3)

    def test_a_star(self):
        grid = [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
        path = a_star(grid, (0, 0), (4, 4))
        self.assertIsNotNone(path)
        self.assertTrue(len(path) > 0)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (4, 4))

    def test_a_star_unreachable(self):
        grid = [
            [0, 1],
            [1, 0],
        ]
        path = a_star(grid, (0, 0), (1, 1))
        self.assertEqual(path, [])


class TestDynamicProgramming(unittest.TestCase):

    def test_fibonacci(self):
        self.assertEqual(fib(0), 0)
        self.assertEqual(fib(1), 1)
        self.assertEqual(fib(5), 5)
        self.assertEqual(fib(10), 55)

    def test_fibonacci_negative(self):
        with self.assertRaises(ValueError):
            fib(-1)

    def test_knapsack(self):
        weights = [2, 3, 4, 5]
        values = [3, 4, 5, 6]
        max_val, chosen = knapsack(weights, values, 5)
        self.assertEqual(max_val, 7)
        self.assertEqual(sum(weights[i] for i in chosen), 5)

    def test_lcs(self):
        self.assertEqual(len(lcs("ABCBDAB", "BDCAB")), 4)
        self.assertEqual(lcs("abc", "def"), "")

    def test_edit_distance(self):
        self.assertEqual(edit_distance("kitten", "sitting"), 3)
        self.assertEqual(edit_distance("", "abc"), 3)
        self.assertEqual(edit_distance("abc", "abc"), 0)

    def test_longest_palindrome(self):
        result = longest_palindrome("babad")
        self.assertIn(result, ["bab", "aba"])

    def test_longest_palindrome_single(self):
        self.assertEqual(longest_palindrome("a"), "a")

    def test_coin_change(self):
        self.assertEqual(coin_change([1, 5, 10, 25], 30), 2)
        self.assertEqual(coin_change([2], 3), -1)

    def test_matrix_chain_order(self):
        dims = [10, 30, 5, 60]
        cost, _ = matrix_chain_order(dims)
        self.assertEqual(cost, 4500)

    def test_rod_cutting(self):
        prices = [1, 5, 8, 9, 10, 17, 17, 20]
        self.assertEqual(rod_cutting(prices, 4), 10)

    def test_subset_sum(self):
        self.assertTrue(subset_sum([3, 34, 4, 12, 5, 2], 9))
        self.assertFalse(subset_sum([3, 34, 4, 12, 5, 2], 30))

    def test_lis(self):
        arr = [10, 9, 2, 5, 3, 7, 101, 18]
        result = lis(arr)
        self.assertEqual(len(result), 4)
        for i in range(1, len(result)):
            self.assertGreater(result[i], result[i - 1])

    def test_lis_nlogn(self):
        arr = [10, 9, 2, 5, 3, 7, 101, 18]
        result = lis_nlogn(arr)
        self.assertEqual(len(result), 4)
        for i in range(1, len(result)):
            self.assertGreater(result[i], result[i - 1])


class TestComplexity(unittest.TestCase):

    def test_measure_complexity(self):
        result = measure_complexity(sorted, [5, 3, 1, 4, 2], repeats=5)
        self.assertIn("avg_time", result)
        self.assertIn("avg_memory", result)
        self.assertGreater(result["avg_time"], 0)

    def test_amortized_analysis(self):
        result = amortized_analysis()
        self.assertIn("n", result)
        self.assertIn("resizes", result)
        self.assertIn("amortized_cost_per_op", result)
        self.assertGreater(result["amortized_cost_per_op"], 0)

    def test_space_complexity(self):
        result = space_complexity(lambda: [i ** 2 for i in range(100)])
        self.assertIn("peak_memory_bytes", result)
        self.assertIn("returned_type", result)
        self.assertEqual(result["returned_type"], "list")


if __name__ == "__main__":
    unittest.main()
