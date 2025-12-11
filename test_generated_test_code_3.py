"""
Auto-generated test cases using RL-based Test Generator
"""
import pytest
from target_code import Stack, Queue

class TestStack:
    def setup_method(self):
        self.instance = Stack()

    # ===== push: 6 tests (from 6 generated cases) =====
    def test_push_case_1(self):
        result = self.instance.push(-1.0)

    def test_push_case_2(self):
        result = self.instance.push(0.0)

    def test_push_case_3(self):
        result = self.instance.push(1.0)

    def test_push_case_4(self):
        result = self.instance.push(2.0)

    def test_push_case_5(self):
        result = self.instance.push(-0.31041693687438965)

    def test_push_case_6(self):
        result = self.instance.push(0.29626983404159546)

    # ===== pop: 1 tests (from 1 generated cases) =====
    def test_pop_exception_1(self):
        with pytest.raises(IndexError):
            self.instance.pop()

    # ===== peek: 1 tests (from 1 generated cases) =====
    def test_peek_exception_1(self):
        with pytest.raises(IndexError):
            self.instance.peek()

    # ===== is_empty: 1 tests (from 1 generated cases) =====
    def test_is_empty_case_1(self):
        result = self.instance.is_empty()
        assert result == True

    # ===== size: 1 tests (from 1 generated cases) =====
    def test_size_case_1(self):
        result = self.instance.size()
        assert result == 0


class TestQueue:
    def setup_method(self):
        self.instance = Queue()

    # ===== enqueue: 6 tests (from 6 generated cases) =====
    def test_enqueue_case_1(self):
        result = self.instance.enqueue(-1.0)

    def test_enqueue_case_2(self):
        result = self.instance.enqueue(0.0)

    def test_enqueue_case_3(self):
        result = self.instance.enqueue(1.0)

    def test_enqueue_case_4(self):
        result = self.instance.enqueue(2.0)

    def test_enqueue_case_5(self):
        result = self.instance.enqueue(0.7031150460243225)

    def test_enqueue_case_6(self):
        result = self.instance.enqueue(0.5695661902427673)

    # ===== dequeue: 1 tests (from 1 generated cases) =====
    def test_dequeue_exception_1(self):
        with pytest.raises(IndexError):
            self.instance.dequeue()

    # ===== is_empty: 1 tests (from 1 generated cases) =====
    def test_is_empty_case_1(self):
        result = self.instance.is_empty()
        assert result == True

    # ===== size: 1 tests (from 1 generated cases) =====
    def test_size_case_1(self):
        result = self.instance.size()
        assert result == 0


