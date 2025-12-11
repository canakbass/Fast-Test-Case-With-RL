"""Test file 3: Data structures"""


class Stack:
    """Simple stack implementation."""
    
    def __init__(self):
        self.items = []
    
    def push(self, item):
        """Add item to stack."""
        self.items.append(item)
    
    def pop(self):
        """Remove and return top item."""
        if self.is_empty():
            raise IndexError("Pop from empty stack")
        return self.items.pop()
    
    def peek(self):
        """Return top item without removing."""
        if self.is_empty():
            raise IndexError("Peek from empty stack")
        return self.items[-1]
    
    def is_empty(self):
        """Check if stack is empty."""
        return len(self.items) == 0
    
    def size(self):
        """Return stack size."""
        return len(self.items)


class Queue:
    """Simple queue implementation."""
    
    def __init__(self):
        self.items = []
    
    def enqueue(self, item):
        """Add item to queue."""
        self.items.insert(0, item)
    
    def dequeue(self):
        """Remove and return first item."""
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")
        return self.items.pop()
    
    def is_empty(self):
        """Check if queue is empty."""
        return len(self.items) == 0
    
    def size(self):
        """Return queue size."""
        return len(self.items)
