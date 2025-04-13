import os

class SolutionParser:
    def __init__(self):
        self.error = None
        self.courses_l_cursor = None

    def _init_state(self, model):
        self.courses_l_cursor = [0] * len(model.courses)
        base_cursor = 0
        for i, course in enumerate(model.courses):
            self.courses_l_cursor[i] = base_cursor
            base_cursor += course.n_lectures

    def _destroy_state(self):
        self.courses_l_cursor = None

    def parse(self, filename, solution):
        if not filename or not os.path.isfile(filename):
            self.error = f"File '{filename}' not found."
            return False

        self._init_state(solution.model)
        try:
            with open(filename, 'r') as f:
                for line_number, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line or line.startswith("//"):
                        continue

                    parts = line.split()
                    if len(parts) != 4:
                        self.error = f"Line {line_number}: Expected 4 fields, got {len(parts)}"
                        return False

                    course_id, room_id, day_str, slot_str = parts

                    course = solution.model.course_by_id.get(course_id)
                    if not course:
                        self.error = f"Line {line_number}: Unknown course '{course_id}'"
                        return False

                    room = solution.model.room_by_id.get(room_id)
                    if not room:
                        self.error = f"Line {line_number}: Unknown room '{room_id}'"
                        return False

                    try:
                        day = int(day_str)
                        slot = int(slot_str)
                    except ValueError:
                        self.error = f"Line {line_number}: Day and Slot must be integers"
                        return False

                    course_idx = course.index
                    lecture_idx = self.courses_l_cursor[course_idx]
                    self.courses_l_cursor[course_idx] += 1

                    if lecture_idx >= len(solution.model.lectures):
                        self.error = f"Line {line_number}: Invalid lecture index for course '{course_id}'"
                        return False

                    solution.assign_lecture(lecture_idx, room.index, day, slot)
        finally:
            self._destroy_state()

        return True

    def get_error(self):
        return self.error
