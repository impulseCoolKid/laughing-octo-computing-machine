import numpy as np

class Node:
    def __init__(self, value):
        self.value = value
        self.next = None
        self.prev = None

class FixedSizeStack:
    def __init__(self, capacity):
        self.capacity = capacity
        self.size = 0
        self.head = None  # Top of the stack
        self.tail = None  # Bottom of the stack

    def push(self, value):
        new_node = Node(value)

        # If stack is empty
        if not self.head:
            self.head = self.tail = new_node
        else:
            # Insert at the head
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node

        self.size += 1

        # Remove the bottom element if over capacity
        if self.size > self.capacity:
            self._remove_tail()

    def _remove_tail(self):
        if self.tail:
            self.tail = self.tail.prev
            if self.tail:
                self.tail.next = None
            self.size -= 1

    def print_stack(self):
        current = self.head
        while current:
            print(current.value, end=" -> ")
            current = current.next
        print("None")

    def to_numpy(self):
        """Return a NumPy array of all node values from top to bottom."""
        values = []
        current = self.head
        while current:
            values.append(current.value)
            current = current.next
        return np.invert(np.array(values))
    
    def containsStim(self):
        all = self.to_numpy()
        for i in range(len(all)):
            if all[i] == 1:
                return 1
        return 0


    def fill_with_zeros(self):
        """
        Artificially fill up the linked list with nodes of value 0
        until the stack's size equals its capacity.
        Nodes are added to the bottom (tail) so that existing nodes remain at the top.
        """
        while self.size < self.capacity:
            new_node = Node(0)
            if not self.head:
                # If the list is empty, set both head and tail to the new node.
                self.head = self.tail = new_node
            else:
                # Append to the tail.
                self.tail.next = new_node
                new_node.prev = self.tail
                self.tail = new_node
            self.size += 1

    def returnStim(self, t, t_dist = 10):
        retdat = None
        current = self.head
        for i in range(t_dist):
            current = current.next
        return current.value


