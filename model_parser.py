from model import Course, Room, Curricula, UnavailabilityConstraint, Teacher, Lecture
from collections import defaultdict

class TimetableModel:
    def __init__(self):
        self.name = None
        self.n_courses = 0
        self.n_rooms = 0
        self.n_days = 0
        self.n_slots = 0
        self.n_curriculas = 0
        self.n_constraints = 0

        self.courses = []
        self.rooms = []
        self.curriculas = []
        self.unavailability_constraints = []

        self.teachers = []
        self.lectures = []

        self.course_by_id = {}
        self.room_by_id = {}
        self.curricula_by_id = {}
        self.teacher_by_id = {}

        self.course_belongs_to_curricula = {}
        self.course_taught_by_teacher = {}
        self.course_availabilities = defaultdict(lambda: True)
        self.curriculas_of_course = defaultdict(list)
        self.courses_of_teacher = defaultdict(list)
        self.courses_of_curricula = defaultdict(list)

    def parse(self, file_path):
        section = None
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("//"):
                    continue

                if line.endswith(":"):
                    section = line[:-1].strip().upper()
                    continue

                if ":" in line:
                    key, value = line.split(':', 1)
                    key, value = key.strip(), value.strip()
                    if key == "Name":
                        self.name = value
                    elif key == "Courses":
                        self.n_courses = int(value)
                    elif key == "Rooms":
                        self.n_rooms = int(value)
                    elif key == "Days":
                        self.n_days = int(value)
                    elif key == "Periods_per_day":
                        self.n_slots = int(value)
                    elif key == "Curricula":
                        self.n_curriculas = int(value)
                    elif key == "Constraints":
                        self.n_constraints = int(value)
                    continue

                # --- Handle each section ---
                if section == "COURSES":
                    parts = line.split()
                    if len(parts) == 5:
                        idx = len(self.courses)
                        course = Course(idx, *parts)
                        self.courses.append(course)
                        self.course_by_id[course.id] = course

                elif section == "ROOMS":
                    parts = line.split()
                    if len(parts) == 2:
                        idx = len(self.rooms)
                        room = Room(idx, *parts)
                        self.rooms.append(room)
                        self.room_by_id[room.id] = room

                elif section == "CURRICULA":
                    parts = line.split()
                    if len(parts) >= 3:
                        idx = len(self.curriculas)
                        curricula_id = parts[0]
                        course_ids = parts[2:]
                        curricula = Curricula(idx, curricula_id, course_ids)
                        self.curriculas.append(curricula)
                        self.curricula_by_id[curricula_id] = curricula
                        for cid in course_ids:
                            self.curriculas_of_course[cid].append(curricula_id)
                            self.courses_of_curricula[curricula_id].append(cid)
                            self.course_belongs_to_curricula[(cid, curricula_id)] = True

                elif section == "UNAVAILABILITY_CONSTRAINTS":
                    parts = line.split()
                    if len(parts) == 3:
                        uc = UnavailabilityConstraint(*parts)
                        self.unavailability_constraints.append(uc)
                        self.course_availabilities[(uc.course_id, uc.day, uc.period)] = False

        self.finalize()

    def finalize(self):
        # Register teachers
        teacher_ids = sorted(set(c.teacher_id for c in self.courses))
        for idx, tid in enumerate(teacher_ids):
            teacher = Teacher(idx, tid)
            self.teachers.append(teacher)
            self.teacher_by_id[tid] = teacher

        # Assign teachers to courses
        for course in self.courses:
            course.teacher = self.teacher_by_id[course.teacher_id]
            self.courses_of_teacher[course.teacher_id].append(course.id)
            self.course_taught_by_teacher[(course.id, course.teacher.id)] = True

        # Attach resolved course to each unavailability constraint
        for uc in self.unavailability_constraints:
            if uc.course_id in self.course_by_id:
                uc.course = self.course_by_id[uc.course_id]

        # Create lectures from courses
        l_index = 0
        for course in self.courses:
            for _ in range(course.n_lectures):
                self.lectures.append(Lecture(l_index, course))
                l_index += 1

    def is_available(self, course_id, day, period):
        return self.course_availabilities.get((course_id, day, period), True)
