import heapq
import random

# TODO fix to consider temporal difference error
# TODO ensure DQN is properly implemented, and consider Rainbow DQN
class PrioritizedReplayBuffer: 
    def __init__(self, capacity=100000): 
        self.heap = []
        self.capacity = capacity

    # TODO fix bug here
    def add(self, experience, priority): 
        if len(self.heap) >= self.capacity: 
            heapq.heappop(self.heap)

        heapq.heappush(self.heap, (-priority, experience))

    # TODO sample based on location in priority queue, instead of randomly
    def sample(self, batch_size): 
        batch = random.choices(self.heap, k=batch_size)

        return zip(*[t[1] for t in batch])        
        
    def len(self): 
        return len(self.heap)