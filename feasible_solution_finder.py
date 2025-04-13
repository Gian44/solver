import math
import random
from collections import defaultdict

class FeasibleSolutionFinderConfig:
    def __init__(self, ranking_randomness=0.33):
        self.ranking_randomness = ranking_randomness


class FeasibleSolutionFinder:
    def __init__(self):
        self.error = None

    def reset(self):
        self.error = None

    def get_course_difficulty(self, model):
        difficulty = [0] * len(model.courses)
        for course in model.courses:
            c = course.index
            t = course.teacher.index

            n_curriculas = len(model.curriculas_of_course[course.id])
            n_teacher_courses = len(model.courses_of_teacher[course.teacher.id])
            n_unavailabilities = sum(
                not model.is_available(course.id, d, s)
                for d in range(model.n_days)
                for s in range(model.n_slots)
            )

            difficulty[c] = (
                n_curriculas +
                n_teacher_courses +
                n_unavailabilities
            ) * max(1, course.n_lectures)
        return difficulty

    def try_find(self, config, solution):
        self.reset()
        model = solution.model
        C, R, D, S = len(model.courses), len(model.rooms), model.n_days, model.n_slots
        Q, T, L = len(model.curriculas), len(model.teachers), len(model.lectures)

        course_difficulty = self.get_course_difficulty(model)

        assignments = []
        for lecture in model.lectures:
            c_idx = lecture.course.index
            r_factor = random.gauss(1, config.ranking_randomness)
            diff = course_difficulty[c_idx] * r_factor
            assignments.append((diff, lecture))
        assignments.sort(reverse=True)

        room_used = [[[False]*S for _ in range(D)] for _ in range(R)]
        teacher_busy = [[[False]*S for _ in range(D)] for _ in range(T)]
        curriculum_used = [[[False]*S for _ in range(D)] for _ in range(Q)]

        n_assignments = 0
        n_attempts = 0

        for _, lecture in assignments:
            c = lecture.course
            c_idx = c.index
            t_idx = c.teacher.index
            curriculas = model.curriculas_of_course[c.id]

            assigned = False
            for r_idx, room in enumerate(model.rooms):
                for d in range(D):
                    for s in range(S):
                        n_attempts += 1
                        if room_used[r_idx][d][s]:
                            continue
                        if teacher_busy[t_idx][d][s]:
                            continue
                        if any(curriculum_used[model.curricula_by_id[q].index][d][s] for q in curriculas):
                            continue
                        if not model.is_available(c.id, d, s):
                            continue

                        solution.assign_lecture(lecture.index, r_idx, d, s)

                        room_used[r_idx][d][s] = True
                        teacher_busy[t_idx][d][s] = True
                        for q in curriculas:
                            q_idx = model.curricula_by_id[q].index
                            curriculum_used[q_idx][d][s] = True

                        n_assignments += 1
                        assigned = True
                        break
                    if assigned: break
                if assigned: break

            if not assigned:
                self.error = f"Failed to assign lecture {lecture.index} (course {c.id})"
                return False

        return True

    def find(self, config, solution, timeout_seconds=None):
        import time
        start_time = time.time()
        trial = 0
        while True:
            if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                break
            if self.try_find(config, solution):
                return True
            solution.clear()
            trial += 1
        return False
