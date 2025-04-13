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
    required_days = model.courses[c1].min_working_days

    # Before
    prev_days = sum(1 for d in range(model.n_days) if sol.sum_cd[c1][d] > 0)

    # After (simulate subtraction of lecture at d1 and addition at d2)
    cur_days = 0
    for d in range(model.n_days):
        count = sol.sum_cd[c1][d]
        if d == d1:
            count -= 1
        if d == d2:
            count += 1
        if count > 0:
            cur_days += 1

    cost = min(0, prev_days - required_days) + max(0, required_days - cur_days)
    return cost * MIN_WORKING_DAYS_COST_FACTOR


def compute_room_stability_cost(sol, c1, r1, c2, r2):
    if c1 < 0 or c1 == c2 or r1 == r2:
        return 0

    model = sol.model

    prev_rooms = sum(1 for r in range(model.n_rooms) if sol.sum_cr[c1][r] > 0)

    cur_rooms = 0
    for r in range(model.n_rooms):
        count = sol.sum_cr[c1][r]
        if r == r1:
            count -= 1
        if r == r2:
            count += 1
        if count > 0:
            cur_rooms += 1

    cost = max(0, cur_rooms - 1) - max(0, prev_rooms - 1)
    return cost * ROOM_STABILITY_COST_FACTOR


def compute_curriculum_compactness_cost(sol, c1, d1, s1, c2, d2, s2):
    if c1 < 0 or c1 == c2:
        return 0

    model = sol.model
    cost = 0

    for q_id in model.curriculas_of_course[model.courses[c1].id]:
        q = model.curricula_by_id[q_id].index

        # skip cost if c1 and c2 share curriculum q
        if c2 >= 0:
            c2_id = model.courses[c2].id
            if c2_id in model.curriculas_of_course and q_id in model.curriculas_of_course[c2_id]:
                continue

        def QDS(q, d, s):
            return 0 <= s < model.n_slots and sol.sum_qds[q][d][s] > 0

        def QDS_OUT_AFTER(q, d, s):
            return not (d == d1 and s == s1) and QDS(q, d, s)

        def QDS_IN_BEFORE(q, d, s):
            return not (d == d1 and s == s1) and QDS(q, d, s)

        def QDS_IN_AFTER(q, d, s):
            return (d == d2 and s == s2) or ((not (d == d1 and s == s1)) and QDS(q, d, s))

        def ALONE_OUT_BEFORE(q, d, s):
            return QDS(q, d, s) and not QDS(q, d, s - 1) and not QDS(q, d, s + 1)

        def ALONE_OUT_AFTER(q, d, s):
            return QDS_OUT_AFTER(q, d, s) and not QDS_OUT_AFTER(q, d, s - 1) and not QDS_OUT_AFTER(q, d, s + 1)

        def ALONE_IN_BEFORE(q, d, s):
            return QDS_IN_BEFORE(q, d, s) and not QDS_IN_BEFORE(q, d, s - 1) and not QDS_IN_BEFORE(q, d, s + 1)

        def ALONE_IN_AFTER(q, d, s):
            return QDS_IN_AFTER(q, d, s) and not QDS_IN_AFTER(q, d, s - 1) and not QDS_IN_AFTER(q, d, s + 1)

        out_prev_cost_before = int(ALONE_OUT_BEFORE(q, d1, s1 - 1))
        out_itself_cost = int(ALONE_OUT_BEFORE(q, d1, s1))
        out_next_cost_before = int(ALONE_OUT_BEFORE(q, d1, s1 + 1))

        out_prev_cost_after = int(ALONE_OUT_AFTER(q, d1, s1 - 1))
        out_next_cost_after = int(ALONE_OUT_AFTER(q, d1, s1 + 1))

        in_prev_cost_before = int(ALONE_IN_BEFORE(q, d2, s2 - 1))
        in_next_cost_before = int(ALONE_IN_BEFORE(q, d2, s2 + 1))
        in_prev_cost_after = int(ALONE_IN_AFTER(q, d2, s2 - 1))
        in_next_cost_after = int(ALONE_IN_AFTER(q, d2, s2 + 1))

        in_itself_cost = int(ALONE_IN_AFTER(q, d2, s2))

        q_cost = (
            (out_prev_cost_after - out_prev_cost_before)
            + (out_next_cost_after - out_next_cost_before)
            + (in_prev_cost_after - in_prev_cost_before)
            + (in_next_cost_after - in_next_cost_before)
            + (in_itself_cost - out_itself_cost)
        )

        cost += q_cost

    return cost * CURRICULUM_COMPACTNESS_COST_FACTOR


def swap_move_compute_cost(sol, mv, result):
    room_cap_1 = compute_room_capacity_cost(sol, mv.helper['c1'], mv.helper['r1'], mv.r2)
    room_cap_2 = compute_room_capacity_cost(sol, mv.helper['c2'], mv.r2, mv.helper['r1'])
    result.delta['room_capacity_cost'] = room_cap_1 + room_cap_2

    min_days_1 = compute_min_working_days_cost(sol, mv.helper['c1'], mv.helper['d1'], mv.helper['c2'], mv.d2)
    min_days_2 = compute_min_working_days_cost(sol, mv.helper['c2'], mv.d2, mv.helper['c1'], mv.helper['d1'])
    result.delta['min_working_days_cost'] = min_days_1 + min_days_2

    room_stab_1 = compute_room_stability_cost(sol, mv.helper['c1'], mv.helper['r1'], mv.helper['c2'], mv.r2)
    room_stab_2 = compute_room_stability_cost(sol, mv.helper['c2'], mv.r2, mv.helper['c1'], mv.helper['r1'])
    result.delta['room_stability_cost'] = room_stab_1 + room_stab_2

    compact_1 = compute_curriculum_compactness_cost(sol, mv.helper['c1'], mv.helper['d1'], mv.helper['s1'],
                                                    mv.helper['c2'], mv.d2, mv.s2)
    compact_2 = compute_curriculum_compactness_cost(sol, mv.helper['c2'], mv.d2, mv.s2,
                                                    mv.helper['c1'], mv.helper['d1'], mv.helper['s1'])
    result.delta['curriculum_compactness_cost'] = compact_1 + compact_2

    result.delta['cost'] = (
        result.delta['room_capacity_cost'] +
        result.delta['min_working_days_cost'] +
        result.delta['room_stability_cost'] +
        result.delta['curriculum_compactness_cost']
    )

    # 🔍 DEBUG PRINT
    print(f"[DELTA DEBUG] RC={room_cap_1 + room_cap_2}, MWD={min_days_1 + min_days_2}, "
          f"RS={room_stab_1 + room_stab_2}, CC={compact_1 + compact_2}, "
          f"Total Delta={result.delta['cost']}")


def swap_predict(sol, mv, require_feasibility=True, compute_cost=True):
    result = SwapResult()
    swap_move_compute_helper(sol, mv)

    if require_feasibility:
        result.feasible = sol.satisfy_hard_constraints_after_swap(mv)

    if compute_cost:
        swap_move_compute_cost(sol, mv, result)

    return result

def swap_extended(sol, mv, strategy='if_feasible_and_better'):
    result = swap_predict(sol, mv, require_feasibility=True, compute_cost=True)

    if strategy == 'always':
        if result.feasible:  # ✔ Add feasibility check here
            swap_move_do(sol, mv)
            return True
    elif strategy == 'if_feasible' and result.feasible:
        swap_move_do(sol, mv)
        return True
    elif strategy == 'if_better' and result.delta['cost'] < 0 and result.feasible:  # ✔ check feasibility
        swap_move_do(sol, mv)
        return True
    elif strategy == 'if_feasible_and_better' and result.feasible and result.delta['cost'] < 0:
        swap_move_do(sol, mv)
        return True

    return False