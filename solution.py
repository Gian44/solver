from collections import namedtuple
import numpy as np

Assignment = namedtuple("Assignment", ["r", "d", "s"])

ROOM_CAPACITY_COST_FACTOR = 1
MIN_WORKING_DAYS_COST_FACTOR = 5
CURRICULUM_COMPACTNESS_COST_FACTOR = 2
ROOM_STABILITY_COST_FACTOR = 1

class Solution:
    def __init__(self, model):
        self.model = model
        self.C = len(model.courses)
        self.R = len(model.rooms)
        self.D = model.n_days
        self.S = model.n_slots
        self.T = len(model.teachers)
        self.Q = len(model.curriculas)
        self.L = len(model.lectures)

        self.assignments = [None] * self.L
        self.timetable_crds = np.zeros((self.C, self.R, self.D, self.S), dtype=int)
        self.sum_cd = np.zeros((self.C, self.D), dtype=int)
        self.sum_cr = np.zeros((self.C, self.R), dtype=int)
        self.sum_qds = np.zeros((self.Q, self.D, self.S), dtype=int)
        self.l_rds = [[[-1 for _ in range(self.S)] for _ in range(self.D)] for _ in range(self.R)]

    def assign_lecture(self, l, r, d, s):
        lecture = self.model.lectures[l]
        c = lecture.course.index
        self.assignments[l] = Assignment(r, d, s)
        self.timetable_crds[c, r, d, s] = 1
        self.sum_cd[c][d] += 1
        self.sum_cr[c][r] += 1
        for q_id in self.model.curriculas_of_course[lecture.course.id]:
            q = self.model.curricula_by_id[q_id].index
            self.sum_qds[q][d][s] += 1
        self.l_rds[r][d][s] = l

    def unassign_lecture(self, l):
        if self.assignments[l] is None:
            return
        r, d, s = self.assignments[l]
        lecture = self.model.lectures[l]
        c = lecture.course.index
        self.assignments[l] = None
        self.timetable_crds[c, r, d, s] = 0
        self.sum_cd[c][d] -= 1
        self.sum_cr[c][r] -= 1
        for q_id in self.model.curriculas_of_course[lecture.course.id]:
            q = self.model.curricula_by_id[q_id].index
            self.sum_qds[q][d][s] -= 1
        self.l_rds[r][d][s] = -1

    def satisfy_hard_constraints_after_swap(self, mv):
        orig_l1 = self.assignments[mv.l1]
        orig_l2 = self.assignments[mv.helper['l2']] if mv.helper['l2'] is not None and mv.helper['l2'] >= 0 else None

        self.unassign_lecture(mv.l1)
        if mv.helper['l2'] is not None and mv.helper['l2'] >= 0:
            self.unassign_lecture(mv.helper['l2'])

        self.assign_lecture(mv.l1, mv.r2, mv.d2, mv.s2)
        if mv.helper['l2'] is not None and mv.helper['l2'] >= 0:
            self.assign_lecture(mv.helper['l2'], mv.helper['r1'], mv.helper['d1'], mv.helper['s1'])

        feasible = self.satisfy_hard_constraints()

        self.unassign_lecture(mv.l1)
        if mv.helper['l2'] is not None and mv.helper['l2'] >= 0:
            self.unassign_lecture(mv.helper['l2'])

        self.assign_lecture(mv.l1, orig_l1.r, orig_l1.d, orig_l1.s)
        if orig_l2:
            self.assign_lecture(mv.helper['l2'], orig_l2.r, orig_l2.d, orig_l2.s)

        return feasible

    def satisfy_hard_constraints(self):
        return (
            self._satisfy_lectures() and
            self._satisfy_room_occupancy() and
            self._satisfy_conflicts() and
            self._satisfy_availabilities()
        )

    def _satisfy_lectures(self):
        for c in range(self.C):
            total = np.sum(self.timetable_crds[c])
            if total != self.model.courses[c].n_lectures:
                return False
            for d in range(self.D):
                for s in range(self.S):
                    if np.sum(self.timetable_crds[c, :, d, s]) > 1:
                        return False
        return True

    def _satisfy_room_occupancy(self):
        for r in range(self.R):
            for d in range(self.D):
                for s in range(self.S):
                    if np.sum(self.timetable_crds[:, r, d, s]) > 1:
                        return False
        return True

    def _satisfy_conflicts(self):
        for q in range(self.Q):
            course_ids = self.model.curriculas[q].courses_ids
            course_indices = [self.model.course_by_id[cid].index for cid in course_ids]
            for d in range(self.D):
                for s in range(self.S):
                    if sum(self.timetable_crds[c, :, d, s].any() for c in course_indices) > 1:
                        return False

        for t in range(self.T):
            teacher = self.model.teachers[t]
            course_ids = self.model.courses_of_teacher[teacher.id]
            course_indices = [self.model.course_by_id[cid].index for cid in course_ids]
            for d in range(self.D):
                for s in range(self.S):
                    if sum(self.timetable_crds[c, :, d, s].any() for c in course_indices) > 1:
                        return False
        return True

    def _satisfy_availabilities(self):
        for c in range(self.C):
            cid = self.model.courses[c].id
            for d in range(self.D):
                for s in range(self.S):
                    if not self.model.is_available(cid, d, s):
                        if self.timetable_crds[c, :, d, s].any():
                            return False
        return True

    def compute_total_cost(self):
        room_capacity_cost = 0
        min_working_days_cost = 0
        curriculum_compactness_cost = 0
        room_stability_cost = 0

        # Room capacity cost
        for c in range(self.C):
            n_students = self.model.courses[c].n_students
            for r in range(self.R):
                for d in range(self.D):
                    for s in range(self.S):
                        if self.timetable_crds[c, r, d, s]:
                            capacity = self.model.rooms[r].capacity
                            if n_students > capacity:
                                room_capacity_cost += (n_students - capacity) * ROOM_CAPACITY_COST_FACTOR

        # Min working days cost
        for c in range(self.C):
            working_days = sum(1 for d in range(self.D) if self.sum_cd[c][d] > 0)
            required_days = self.model.courses[c].min_working_days
            if working_days < required_days:
                min_working_days_cost += (required_days - working_days) * MIN_WORKING_DAYS_COST_FACTOR

        # Curriculum compactness cost
        for q in range(self.Q):
            for d in range(self.D):
                for s in range(self.S):
                    if self.sum_qds[q][d][s] > 0:
                        neighbors = 0
                        if s > 0:
                            neighbors += self.sum_qds[q][d][s - 1]
                        if s + 1 < self.S:
                            neighbors += self.sum_qds[q][d][s + 1]
                        if neighbors == 0:
                            curriculum_compactness_cost += CURRICULUM_COMPACTNESS_COST_FACTOR

        # Room stability cost
        for c in range(self.C):
            room_count = sum(1 for r in range(self.R) if self.sum_cr[c][r] > 0)
            if room_count > 1:
                room_stability_cost += (room_count - 1) * ROOM_STABILITY_COST_FACTOR

        total_cost = (
            room_capacity_cost +
            min_working_days_cost +
            curriculum_compactness_cost +
            room_stability_cost
        )

        # print(f"[COST BREAKDOWN] RoomCapacity: {room_capacity_cost}, "
        #     f"MinWorkingDays: {min_working_days_cost}, "
        #     f"CurriculumCompactness: {curriculum_compactness_cost}, "
        #     f"RoomStability: {room_stability_cost} => Total: {total_cost}")

        return total_cost


    def copy_from(self, other):
        for l, a in enumerate(other.assignments):
            if a is not None:
                self.assign_lecture(l, a.r, a.d, a.s)

    def to_string(self):
        output = []
        for l, a in enumerate(self.assignments):
            if a is not None:
                course = self.model.lectures[l].course.id
                room = self.model.rooms[a.r].id
                output.append(f"{course} {room} {a.d} {a.s}")
        return "\n".join(output)
    
    def sum_qds_for_course(self, c, d, s, delta):
        course_id = self.model.courses[c].id
        for q_id in self.model.curriculas_of_course[course_id]:
            q = self.model.curricula_by_id[q_id].index
            self.sum_qds[q][d][s] += delta

