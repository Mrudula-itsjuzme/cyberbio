class BudgetManager:
    def __init__(self, max_queries: int = 100, max_generations: int = 500, max_runtime_seconds: float = 60.0):
        self.max_queries = max_queries
        self.max_generations = max_generations
        self.max_runtime_seconds = max_runtime_seconds
        
        self.queries_used = 0
        self.generations_used = 0
        self._start_time = None

    def start(self):
        import time
        self._start_time = time.time()
        self.queries_used = 0
        self.generations_used = 0

    def record_query(self, count: int = 1):
        self.queries_used += count

    def record_generation(self, count: int = 1):
        self.generations_used += count

    @property
    def runtime_seconds(self) -> float:
        if self._start_time is None:
            return 0.0
        import time
        return time.time() - self._start_time

    def is_exhausted(self) -> bool:
        if self.queries_used >= self.max_queries:
            return True
        if self.generations_used >= self.max_generations:
            return True
        if self.max_runtime_seconds > 0 and self.runtime_seconds >= self.max_runtime_seconds:
            return True
        return False
