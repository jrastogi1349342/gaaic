import heapq
import random

class PrioritizedReplayBuffer: 
    def __init__(self, capacity=100000): 
        self.heap = []
        self.capacity = capacity

    def add(self, experience, priority): 
        if len(self.heap) >= self.capacity: 
            heapq.heappop(self.heap)

        heapq.heappush(self.heap, (-priority, experience))

    def sample(self, batch_size): 
        batch = random.choices(self.heap, k=batch_size)

        return zip(*[t[1] for t in batch])        
        