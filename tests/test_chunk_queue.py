from __future__ import annotations

import unittest

import numpy as np

from roboeval_entry.control.chunk import ActionChunkQueue


class ChunkQueueTests(unittest.TestCase):
    def test_truncates_to_execute_steps(self):
        q = ActionChunkQueue(execute_steps=8)
        q.push(np.zeros((16, 14), dtype=np.float32))
        self.assertEqual(q.remaining, 8)

    def test_pop_order_and_empty_raise(self):
        q = ActionChunkQueue(2)
        q.push(np.tile(np.arange(14), (3, 1)).astype(np.float32))
        np.testing.assert_array_equal(q.pop(), np.arange(14, dtype=np.float32))
        q.pop()
        with self.assertRaises(RuntimeError):
            q.pop()

    def test_reset_empties(self):
        q = ActionChunkQueue(4)
        q.push(np.zeros((4, 14), dtype=np.float32))
        q.reset()
        self.assertEqual(q.remaining, 0)
        with self.assertRaises(RuntimeError):
            q.pop()

    def test_short_chunk_rejected(self):
        q = ActionChunkQueue(8)
        with self.assertRaises(ValueError):
            q.push(np.zeros((4, 14), dtype=np.float32))

    def test_bad_shape_rejected(self):
        q = ActionChunkQueue(4)
        with self.assertRaises(ValueError):
            q.push(np.zeros((4, 10), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
