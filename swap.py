import random
from utils.indexing import INDEX2, INDEX3
from solution import ROOM_CAPACITY_COST_FACTOR, MIN_WORKING_DAYS_COST_FACTOR, CURRICULUM_COMPACTNESS_COST_FACTOR, ROOM_STABILITY_COST_FACTOR

class SwapMove:
    def __init__(self, l1, r2, d2, s2):
        self.l1 = l1
        self.r2 = r2
        self.d2 = d2
        self.s2 = s2
        self.helper = {
            'c1': None, 'r1': None, 'd1': None, 's1': None,
            'l2': None, 'c2': None
        }

class SwapResult:
    def __init__(self):
        self.feasible = False
        self.delta = {
            'room_capacity_cost': 0,
            'min_working_days_cost': 0,
            'curriculum_compactness_cost': 0,
            'room_stability_cost': 0,
            'cost': 0
        }

def swap_move_compute_helper(sol, mv):
    l1 = mv.l1
    lecture = sol.model.lectures[l1]
    a = sol.assignments[l1]
    mv.helper['c1'] = lecture.course.index
    mv.helper['r1'] = a.r
    mv.helper['d1'] = a.d
    mv.helper['s1'] = a.s

    mv.helper['l2'] = sol.l_rds[mv.r2][mv.d2][mv.s2]
    if mv.helper['l2'] >= 0:
        mv.helper['c2'] = sol.model.lectures[mv.helper['l2']].course.index
    else:
        mv.helper['c2'] = -1

def swap_move_is_effective(mv):
    return mv.helper['c1'] != mv.helper['c2']

def swap_move_do(sol, mv):
    assert sol.assignments[mv.l1].r == mv.helper['r1']
    assert sol.assignments[mv.l1].d == mv.helper['d1']
    assert sol.assignments[mv.l1].s == mv.helper['s1']

    sol.unassign_lecture(mv.l1)
    if mv.helper['l2'] >= 0:
        sol.unassign_lecture(mv.helper['l2'])

    sol.assign_lecture(mv.l1, mv.r2, mv.d2, mv.s2)
    if mv.helper['l2'] >= 0:
        sol.assign_lecture(mv.helper['l2'], mv.helper['r1'], mv.helper['d1'], mv.helper['s1'])

def compute_room_capacity_cost(sol, c1, r1, r2):
    if c1 < 0:
        return 0
    students = sol.model.courses[c1].n_students
    cap1 = sol.model.rooms[r1].capacity
    cap2 = sol.model.rooms[r2].capacity
    cost = min(0, cap1 - students) + max(0, students - cap2)
    return cost * ROOM_CAPACITY_COST_FACTOR

def compute_min_working_days_cost(sol, c1, d1, c2, d2):
    if c1 < 0 or c1 == c2:
        return 0

    model = sol.model
    prev_days = sum(1 for d in range(model.n_days) if sol.sum_cd[c1][d] > 0)

    delta_cd = sol.sum_cd[c1].copy()
    delta_cd[d1] -= 1
    delta_cd[d2] += 1
    curr_days = sum(1 for d in range(model.n_days) if delta_cd[d] > 0)

    required = model.courses[c1].min_working_days
    prev_cost = max(0, required - prev_days)
    curr_cost = max(0, required - curr_days)
    
    return (curr_cost - prev_cost) * MIN_WORKING_DAYS_COST_FACTOR

def compute_room_stability_cost(sol, c1, r1, c2, r2):
    if c1 < 0 or r1 == r2 or c1 == c2:
        return 0

    model = sol.model
    before_rooms = sum(1 for r in sol.sum_cr[c1] if r > 0)

    simulated = sol.sum_cr[c1].copy()
    simulated[r1] -= 1
    simulated[r2] += 1
    after_rooms = sum(1 for r in simulated if r > 0)

    cost = max(0, after_rooms - 1) - max(0, before_rooms - 1)
    return cost * ROOM_STABILITY_COST_FACTOR

def compute_curriculum_compactness_cost(sol, c1, d1, s1, c2, d2, s2):
    if c1 < 0 or c1 == c2:
        return 0

    model = sol.model
    delta_cost = 0

    for q_id in model.curriculas_of_course[model.courses[c1].id]:
        q = model.curricula_by_id[q_id].index

        if c2 >= 0:
            c2_id = model.courses[c2].id
            if c2_id in model.curriculas_of_course and q_id in model.curriculas_of_course[c2_id]:
                continue

        sum_qds = sol.sum_qds[q].copy()
        sum_qds[d1][s1] -= 1
        sum_qds[d2][s2] += 1

        def is_isolated(d, s):
            if not (0 <= s < model.n_slots):
                return False
            if sum_qds[d][s] <= 0:
                return False
            left = sum_qds[d][s - 1] if s > 0 else 0
            right = sum_qds[d][s + 1] if s + 1 < model.n_slots else 0
            return left == 0 and right == 0

        before = is_isolated(d1, s1)
        after = is_isolated(d2, s2)
        delta_cost += int(after) - int(before)

    return delta_cost * CURRICULUM_COMPACTNESS_COST_FACTOR

def swap_move_compute_cost(sol, mv, result):
    simulate_swap_delta(sol, mv)

    result.delta['room_capacity_cost'] = (
        compute_room_capacity_cost(sol, mv.helper['c1'], mv.helper['r1'], mv.r2) +
        compute_room_capacity_cost(sol, mv.helper['c2'], mv.r2, mv.helper['r1'])
    )

    result.delta['min_working_days_cost'] = (
        compute_min_working_days_cost(sol, mv.helper['c1'], mv.helper['d1'], mv.helper['c2'], mv.d2) +
        compute_min_working_days_cost(sol, mv.helper['c2'], mv.d2, mv.helper['c1'], mv.helper['d1'])
    )

    result.delta['room_stability_cost'] = (
        compute_room_stability_cost(sol, mv.helper['c1'], mv.helper['r1'], mv.helper['c2'], mv.r2) +
        compute_room_stability_cost(sol, mv.helper['c2'], mv.r2, mv.helper['c1'], mv.helper['r1'])
    )

    result.delta['curriculum_compactness_cost'] = (
        compute_curriculum_compactness_cost(sol, mv.helper['c1'], mv.helper['d1'], mv.helper['s1'],
                                            mv.helper['c2'], mv.d2, mv.s2) +
        compute_curriculum_compactness_cost(sol, mv.helper['c2'], mv.d2, mv.s2,
                                            mv.helper['c1'], mv.helper['d1'], mv.helper['s1'])
    )

    result.delta['cost'] = sum(result.delta.values())

    # print(f"[DELTA DEBUG] RC={result.delta['room_capacity_cost']}, "
    #       f"MWD={result.delta['min_working_days_cost']}, "
    #       f"RS={result.delta['room_stability_cost']}, "
    #       f"CC={result.delta['curriculum_compactness_cost']}, "
    #       f"Total Delta={result.delta['cost']}")

    undo_simulate_swap(sol, mv)

def simulate_swap_delta(sol, mv):
    c1 = mv.helper['c1']
    sol.sum_cd[c1][mv.helper['d1']] -= 1
    sol.sum_cr[c1][mv.helper['r1']] -= 1
    sol.l_rds[mv.helper['r1']][mv.helper['d1']][mv.helper['s1']] = -1
    sol.sum_qds_for_course(c1, mv.helper['d1'], mv.helper['s1'], -1)

    if mv.helper['l2'] is not None and mv.helper['l2'] >= 0:
        c2 = mv.helper['c2']
        sol.sum_cd[c2][mv.d2] -= 1
        sol.sum_cr[c2][mv.r2] -= 1
        sol.l_rds[mv.r2][mv.d2][mv.s2] = -1
        sol.sum_qds_for_course(c2, mv.d2, mv.s2, -1)

    sol.sum_cd[c1][mv.d2] += 1
    sol.sum_cr[c1][mv.r2] += 1
    sol.l_rds[mv.r2][mv.d2][mv.s2] = mv.l1
    sol.sum_qds_for_course(c1, mv.d2, mv.s2, +1)

    if mv.helper['l2'] is not None and mv.helper['l2'] >= 0:
        sol.sum_cd[c2][mv.helper['d1']] += 1
        sol.sum_cr[c2][mv.helper['r1']] += 1
        sol.l_rds[mv.helper['r1']][mv.helper['d1']][mv.helper['s1']] = mv.helper['l2']
        sol.sum_qds_for_course(c2, mv.helper['d1'], mv.helper['s1'], +1)

def undo_simulate_swap(sol, mv):
    c1 = mv.helper['c1']
    sol.sum_cd[c1][mv.d2] -= 1
    sol.sum_cr[c1][mv.r2] -= 1
    sol.l_rds[mv.r2][mv.d2][mv.s2] = -1
    sol.sum_qds_for_course(c1, mv.d2, mv.s2, -1)

    sol.sum_cd[c1][mv.helper['d1']] += 1
    sol.sum_cr[c1][mv.helper['r1']] += 1
    sol.l_rds[mv.helper['r1']][mv.helper['d1']][mv.helper['s1']] = mv.l1
    sol.sum_qds_for_course(c1, mv.helper['d1'], mv.helper['s1'], +1)

    if mv.helper['l2'] is not None and mv.helper['l2'] >= 0:
        c2 = mv.helper['c2']
        sol.sum_cd[c2][mv.helper['d1']] -= 1
        sol.sum_cr[c2][mv.helper['r1']] -= 1
        sol.l_rds[mv.helper['r1']][mv.helper['d1']][mv.helper['s1']] = -1
        sol.sum_qds_for_course(c2, mv.helper['d1'], mv.helper['s1'], -1)

        sol.sum_cd[c2][mv.d2] += 1
        sol.sum_cr[c2][mv.r2] += 1
        sol.l_rds[mv.r2][mv.d2][mv.s2] = mv.helper['l2']
        sol.sum_qds_for_course(c2, mv.d2, mv.s2, +1)

def swap_predict(sol, mv, require_feasibility=True, compute_cost=True):
    swap_move_compute_helper(sol, mv)
    result = SwapResult()

    if require_feasibility:
        result.feasible = sol.satisfy_hard_constraints_after_swap(mv)

    if compute_cost:
        swap_move_compute_cost(sol, mv, result)

    return result

def swap_extended(sol, mv, strategy='if_feasible_and_better'):
    swap_move_compute_helper(sol, mv)
    result = swap_predict(sol, mv, require_feasibility=True, compute_cost=True)

    if strategy == 'always':
        if result.feasible:
            swap_move_do(sol, mv)
            return True
    elif strategy == 'if_feasible' and result.feasible:
        swap_move_do(sol, mv)
        return True
    elif strategy == 'if_better' and result.delta['cost'] < 0 and result.feasible:
        swap_move_do(sol, mv)
        return True
    elif strategy == 'if_feasible_and_better' and result.feasible and result.delta['cost'] < 0:
        swap_move_do(sol, mv)
        return True

    return False
