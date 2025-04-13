from collections import defaultdict

class Course:
    def __init__(self, index, course_id, teacher_id, n_lectures, min_working_days, n_students):
        self.index = index
        self.id = course_id
        self.teacher_id = teacher_id
        self.n_lectures = int(n_lectures)
        self.min_working_days = int(min_working_days)
        self.n_students = int(n_students)
        self.teacher = None  # Will be assigned in finalize()

class Room:
    def __init__(self, index, room_id, capacity):
        self.index = index
        self.id = room_id
        self.capacity = int(capacity)

class Curricula:
    def __init__(self, index, curricula_id, course_ids):
        self.index = index
        self.id = curricula_id
        self.courses_ids = course_ids
        self.n_courses = len(course_ids)

class UnavailabilityConstraint:
    def __init__(self, course_id, day, period):
        self.course_id = course_id
        self.day = int(day)
        self.period = int(period)
        self.course = None  # Will be resolved to pointer during finalize()

class Teacher:
    def __init__(self, index, teacher_id):
        self.index = index
        self.id = teacher_id

class Lecture:
    def __init__(self, index, course):
        self.index = index
        self.course = course
