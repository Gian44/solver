import random
from swap import SwapMove, swap_predict, swap_extended

class HeuristicSolverState:
    def __init__(self, model, current_solution, best_solution,
                 current_cost=float('inf'), best_cost=float('inf'),
                 cycle=0, method=0,
                 non_improving_best_cycles=0, non_improving_current_cycles=0,
                 config=None, stats=None):
        self.model = model
        self.current_solution = current_solution
        self.best_solution = best_solution

        self.current_cost = current_cost
        self.best_cost = best_cost
        self.cycle = cycle
        self.method = method
        self.non_improving_best_cycles = non_improving_best_cycles
        self.non_improving_current_cycles = non_improving_current_cycles
        self._last_log_time = 0
        self.config = config
        self.stats = stats

        self.L = len(model.lectures)
        self.R = model.n_rooms
        self.D = model.n_days
        self.S = model.n_slots

        self.methods_name = []
        if self.stats:
            self.stats.methods = []
            self.stats.n_methods = 0
            self.stats.best_solution_time = 0
            self.stats.starting_time = 0
            self.stats.ending_time = 0
            self.stats.move_count = 0
            self.stats.cycle_count = 0
            self.stats.best_restored_count = 0

    def update_best_solution(self):
        improved = False
        if self.current_cost < self.best_cost:
            print(f"[Update] New best found with cost {self.current_cost}")
            delta = self.current_cost - self.best_cost if self.best_cost != float('inf') else 0
            self.best_cost = self.current_cost
            self.best_solution.copy_from(self.current_solution)
            improved = True

            if self.stats:
                self.stats.best_solution_time = 0  # Replace with time.monotonic() if tracking time
                self.stats.methods[self.method]['improvement_count'] += 1
                self.stats.methods[self.method]['improvement_delta'] += delta

        if self.stats:
            self.stats.move_count += 1
            self.stats.methods[self.method]['move_count'] += 1

        return improved

    
    def predict_swap_cost(self, mv, require_feasibility=False):
        return swap_predict(self.current_solution, mv, require_feasibility=require_feasibility, compute_cost=True)

    def apply_swap(self, mv):
        # Perform the swap and calculate the cost before and after the swap
        current_cost_before = self.current_solution.compute_total_cost()

        applied = swap_extended(self.current_solution, mv, strategy='always')

        if applied:
            # Calculate the new cost after the swap
            current_cost_after = self.current_solution.compute_total_cost()
            # Update the current cost based on the difference between the new and old cost
            self.current_cost = current_cost_after

        return applied


    def generate_swap_move(self):
        while True:
            l1, r2, d2, s2 = generate_random_lecture_slot(self.model)
            mv = SwapMove(l1, r2, d2, s2)
            swap_predict(self.current_solution, mv, require_feasibility=False, compute_cost=False)
            if mv.helper['c1'] != mv.helper['c2']:
                return mv

def generate_random_lecture_slot(model):
    l = random.randint(0, len(model.lectures) - 1)
    r = random.randint(0, model.n_rooms - 1)
    d = random.randint(0, model.n_days - 1)
    s = random.randint(0, model.n_slots - 1)
    return l, r, d, s

