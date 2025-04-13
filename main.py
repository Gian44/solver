# main.py
from model_parser import TimetableModel  # You need to have a parser for your ITC2007 input files
from solution import *
from feasible_solution_finder import FeasibleSolutionFinder, FeasibleSolutionFinderConfig
from heuristic_solver_state import HeuristicSolverState
from simulated_annealing import SimulatedAnnealingParams, simulated_annealing

import time
from config import *


def timeout_callback_factory(seconds):
    start = time.time()
    return lambda: time.time() - start > seconds


def verbose_callback(iteration, idle, current, local_best, global_best, temperature):
    print(f"Iter {iteration} | Idle {idle} | Curr {current} | Local Best {local_best} "
          f"| Global Best {global_best} | Temp {temperature:.4f}")


def main():
    # === Load model ===
    model = TimetableModel()
    model.parse(INPUT)
    print("Model loaded.")

    # === Prepare initial feasible solution ===
    finder = FeasibleSolutionFinder()
    config = FeasibleSolutionFinderConfig()
    solution = Solution(model)

    print("Finding initial feasible solution...")
    if not finder.find(config, solution):
        print("Failed to find an initial feasible solution.")
        return

    initial_cost = solution.compute_total_cost()
    print("Initial feasible solution cost:", initial_cost)

    # === Prepare solver state ===
    best_solution = Solution(model)
    best_solution.copy_from(solution)

    state = HeuristicSolverState(model=model,
                                  current_solution=solution,
                                  best_solution=best_solution,
                                  current_cost=initial_cost,
                                  best_cost=initial_cost)
    
    state.methods_name = ["Simulated Annealing"]
    state.method = 0


    # === Configure Simulated Annealing ===
    params = SimulatedAnnealingParams(
        initial_temperature=1.4,
        cooling_rate=0.965,
        temperature_length_coeff=0.125,
        min_temperature=0.12,
        min_temperature_near_best_coeff=0.68,
        near_best_ratio=1.05,
        reheat_coeff=1.015
    )

    # === Run Simulated Annealing ===
    print("Running Simulated Annealing...")
    timeout = timeout_callback_factory(60)  # 60-second timeout
    simulated_annealing(state, params, timeout_callback=timeout, verbose_callback=verbose_callback)

    print("\nFinal best cost:", state.best_cost)
    print("Final best solution:")
    print(state.best_solution.to_string())

    with open(OUTPUT, "w") as f:
        f.write(state.best_solution.to_string())

    


if __name__ == "__main__":
    main()
