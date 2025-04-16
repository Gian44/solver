import math
import random

class SimulatedAnnealingParams:
    def __init__(self,
                 initial_temperature=1.4,
                 cooling_rate=0.965,
                 temperature_length_coeff=0.125,
                 min_temperature=0.12,
                 min_temperature_near_best_coeff=0.68,
                 near_best_ratio=1.05,
                 reheat_coeff=1.015):
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.temperature_length_coeff = temperature_length_coeff
        self.min_temperature = min_temperature
        self.min_temperature_near_best_coeff = min_temperature_near_best_coeff
        self.near_best_ratio = near_best_ratio
        self.reheat_coeff = reheat_coeff

def sa_acceptance(delta, temperature):
    return math.exp(-delta / temperature)

def simulated_annealing(state, params, timeout_callback=None, verbose_callback=None):
    model = state.model
    t_len = int(state.L * state.R * state.D * state.S * params.temperature_length_coeff)

    t = params.initial_temperature
    t_min = params.min_temperature
    t_min_near_best = t_min * params.min_temperature_near_best_coeff
    cooling_rate = params.cooling_rate

    reheat = params.reheat_coeff ** state.non_improving_best_cycles
    t *= reheat

    local_best_cost = state.current_cost
    idle = 0
    iter_count = 0

    while True:
        if timeout_callback and timeout_callback():
            break

        is_near_best = state.current_cost < round(params.near_best_ratio * state.best_cost)
        if (is_near_best and t <= t_min_near_best) or t <= t_min:
            break

        for _ in range(t_len):
            mv = state.generate_swap_move()

            # Store the previous cost
            prev_cost = state.current_cost

            # Apply the swap
            state.apply_swap(mv)

            # Calculate the new cost after the swap
            new_cost = state.current_solution.compute_total_cost()

            # Compute the delta as the difference between the current and previous cost
            delta_cost = new_cost - prev_cost

            # Accept the move if the new cost is lower or based on the acceptance probability
            accept = (new_cost < state.best_cost or random.random() < sa_acceptance(delta_cost, t))

            if accept:
                # Update the current cost after applying the swap
                state.current_cost = new_cost

                # 🔍 DEBUG: Validate if delta cost matches recomputed cost
                actual_cost = state.current_solution.compute_total_cost()
                if state.current_cost != actual_cost:
                    print(f"[DEBUG MISMATCH] Estimated Cost: {state.current_cost}, Actual Cost: {actual_cost}")
                    print(f"Delta Applied: {delta_cost}, Previous Cost: {prev_cost}")

                state.update_best_solution()

            if state.current_cost < local_best_cost:
                local_best_cost = state.current_cost
                idle = 0
            else:
                idle += 1

            if verbose_callback and ((idle > 0 and idle % 10000 == 0) or iter_count % (5 * t_len) == 0):
                verbose_callback(iter_count, idle, state.current_cost, local_best_cost,
                                 state.best_cost, t)

            iter_count += 1

        t *= cooling_rate
