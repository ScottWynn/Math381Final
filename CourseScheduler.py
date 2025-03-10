import numpy as np
import matplotlib.pyplot as plt
import random
import pulp as pl
from datetime import datetime, timedelta

#Dataset Definitions. Note a different random seed is used for each different randomized variable
rooms = ["R1", "R2", "R3", "R4"]
courses = [f"C{i}" for i in range(1, 31)]
class_cap = {rooms[i]: cap for i, cap in enumerate([100, 80, 60, 50])}
#Course capacity
np.random.seed(42)
course_cap = {courses[i]: cap for i, cap in enumerate(np.random.randint(10, 100, size=len(courses)))}
#Start times
np.random.seed(123)
course_start_minutes = {c: np.random.randint(0, 540) for c in courses}  # 0 to 540 minutes (8:00 to 17:00)
def minutes_to_timestr(minutes):
    hour = 8 + minutes // 60
    minute = minutes % 60
    return f"{hour:02d}:{minute:02d}"


course_start_times = {c: minutes_to_timestr(m) for c, m in course_start_minutes.items()}
#Whiteboards and Projectors (rooms)
np.random.seed(234)
has_whiteboard = {room: np.random.choice([True, False]) for room in rooms}
np.random.seed(345)
has_projector = {room: np.random.choice([True, False]) for room in rooms}
# Whiteboards and projector requirements (classes)
np.random.seed(456)
requires_whiteboard = {course: np.random.choice([True, False]) for course in courses}
np.random.seed(567)
requires_projector = {course: np.random.choice([True, False]) for course in courses}

# Time resolution intervals (5 mins)
start_time = datetime.strptime("08:00", "%H:%M")
end_time = datetime.strptime("17:00", "%H:%M")
time_interval = timedelta(minutes=5)
time_slots = []
current_time = start_time
while current_time <= end_time:
    time_slots.append(current_time.strftime("%H:%M"))
    current_time += time_interval
course_duration = 12  # 1 hour = 12 × 5-minute intervals

#Linear model
model = pl.LpProblem("Classroom_Scheduling", pl.LpMaximize)

# Setting decision variables: x[c,r] = 1 if course c is assigned to room r and objective function
x = pl.LpVariable.dicts("assignment",
                        [(c, r) for c in courses for r in rooms],
                        cat='Binary')
model += pl.lpSum([course_cap[c] * x[(c, r)] for c in courses for r in rooms])

# Constraints
for c in courses:
    model += pl.lpSum([x[(c, r)] for r in rooms]) <= 1
for c in courses:
    for r in rooms:
        model += course_cap[c] * x[(c, r)] <= class_cap[r]
for i, c1 in enumerate(courses):
    start1 = course_start_minutes[c1]
    #Duration per course - replace with randomized if desired
    end1 = start1 + 60
    for c2 in courses[i + 1:]:
        start2 = course_start_minutes[c2]
        end2 = start2 + 60
        if max(start1, start2) < min(end1, end2):  # Overlap condition
            for r in rooms:
                model += x[(c1, r)] + x[(c2, r)] <= 1
for c in courses:
    for r in rooms:
        # If course requires whiteboard but room doesn't have one, prevent assignment
        if requires_whiteboard[c] and not has_whiteboard[r]:
            model += x[(c, r)] == 0
for c in courses:
    for r in rooms:
        # If course requires projector but room doesn't have one, prevent assignment
        if requires_projector[c] and not has_projector[r]:
            model += x[(c, r)] == 0

model.solve(pl.PULP_CBC_CMD(msg=False))

#Results and Printing Stats
schedule = {}
for c in courses:
    for r in rooms:
        if pl.value(x[(c, r)]) > 0.5:
            schedule[c] = (r, course_start_times[c])
print(f"ILP Solution: Scheduled {len(schedule)} out of {len(courses)} courses")
total_students_ilp = sum(course_cap[c] for c in schedule)
print(f"Total students accommodated: {total_students_ilp}")
print("\nRoom resources:")
for r in rooms:
    resources = []
    if has_whiteboard[r]: resources.append("WB")
    if has_projector[r]: resources.append("PR")
    resources_str = ", ".join(resources) if resources else "None"
    print(f"{r}: {resources_str}")
def time_to_float(time_str):
    hour, minute = map(int, time_str.split(':'))
    return hour + minute / 60


#Plotting stuff
def get_course_color(course):
    wb = requires_whiteboard[course]
    pr = requires_projector[course]
    if wb and pr:
        return 'purple'
    elif wb:
        return 'cornflowerblue'
    elif pr:
        return 'orange'
    else:
        return 'lightblue'

plt.figure(figsize=(14, 8))
room_indices = {room: i for i, room in enumerate(rooms)}

for course, (room, start_time_str) in schedule.items():
    room_idx = room_indices[room]
    start_float = time_to_float(start_time_str)

    color = get_course_color(course)
    plt.barh(room_idx, 1, left=start_float, align='center',
             color=color, edgecolor='black', alpha=0.7)

    resources = []
    if requires_whiteboard[course]: resources.append("WB")
    if requires_projector[course]: resources.append("PR")
    resources_str = ",".join(resources) if resources else ""

    plt.text(start_float + 0.5, room_idx,
             f"{course}\n{start_time_str}\n({course_cap[course]}) {resources_str}",
             va='center', ha='center', fontsize=8, fontweight='bold')

room_labels = []
for r in rooms:
    resources = []
    if has_whiteboard[r]: resources.append("WB")
    if has_projector[r]: resources.append("PR")
    resources_str = ",".join(resources) if resources else ""
    room_labels.append(f"{r} (cap:{class_cap[r]}) {resources_str}")

plt.yticks(range(len(rooms)), room_labels)
plt.xticks(range(8, 19), [f"{h}:00" for h in range(8, 19)])
plt.xlim(7.75, 18.25)
plt.xlabel("Time")
plt.ylabel("Classrooms")
plt.title(
    f"Optimized Classroom Schedule with Resource Constraints\n{len(schedule)}/{len(courses)} courses scheduled, {total_students_ilp} students accommodated")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()

#Utilization stats
total_minutes = 9 * 60  # 9 hours (8:00-17:00)
total_room_minutes = len(rooms) * total_minutes
scheduled_minutes = len(schedule) * 60  # Each course is 60 minutes
time_utilization = scheduled_minutes / total_room_minutes

max_possible_students = sum(class_cap[r] * (total_minutes / 60) for r in rooms)
student_utilization = total_students_ilp / max_possible_students

print(f"Time slot utilization: {time_utilization:.2%}")
print(f"Student capacity utilization: {student_utilization:.2%}")


#Naive Solution and stats

def naive_scheduling():
    sorted_courses = sorted(courses, key=lambda c: course_start_minutes[c])
    room_schedules = {r: [] for r in rooms}
    naive_schedule = {}

    for course in sorted_courses:
        start_time = course_start_minutes[course]
        end_time = start_time + 60  # 1 hour duration
        course_size = course_cap[course]
        needs_whiteboard = requires_whiteboard[course]
        needs_projector = requires_projector[course]

        for room in rooms:
            if class_cap[room] < course_size:
                continue
            if needs_whiteboard and not has_whiteboard[room]:
                continue
            if needs_projector and not has_projector[room]:
                continue
            conflict = False
            for (existing_course, existing_start, existing_end) in room_schedules[room]:
                if max(start_time, existing_start) < min(end_time, existing_end):
                    conflict = True
                    break

            if not conflict:
                room_schedules[room].append((course, start_time, end_time))
                naive_schedule[course] = (room, course_start_times[course])
                break

    return naive_schedule


# Run naive scheduling, print and plot results
naive_schedule = naive_scheduling()
print("\nNaive Solution: Scheduled {0} out of {1} courses".format(
    len(naive_schedule), len(courses)))
total_students_naive = sum(course_cap[c] for c in naive_schedule)
print("Total students accommodated: {0}".format(total_students_naive))

def get_naive_course_color(course):
    wb = requires_whiteboard[course]
    pr = requires_projector[course]
    if wb and pr:
        return 'darkviolet'
    elif wb:
        return 'forestgreen'
    elif pr:
        return 'darkorange'
    else:
        return 'lightgreen'


plt.figure(figsize=(14, 8))

for course, (room, start_time_str) in naive_schedule.items():
    room_idx = room_indices[room]
    start_float = time_to_float(start_time_str)
    color = get_naive_course_color(course)
    plt.barh(room_idx, 1, left=start_float, align='center',
             color=color, edgecolor='black', alpha=0.7)

    resources = []
    if requires_whiteboard[course]: resources.append("WB")
    if requires_projector[course]: resources.append("PR")
    resources_str = ",".join(resources) if resources else ""

    plt.text(start_float + 0.5, room_idx,
             f"{course}\n{start_time_str}\n({course_cap[course]}) {resources_str}",
             va='center', ha='center', fontsize=8, fontweight='bold')

plt.yticks(range(len(rooms)), room_labels)
plt.xticks(range(8, 19), [f"{h}:00" for h in range(8, 19)])
plt.xlim(7.75, 18.25)
plt.xlabel("Time")
plt.ylabel("Classrooms")
plt.title(
    f"Naive Classroom Schedule with Resource Constraints\n{len(naive_schedule)}/{len(courses)} courses scheduled, {total_students_naive} students accommodated")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()

#Improvement Metrics of ILP over the Naive Solution
students_improvement = total_students_ilp - total_students_naive
percent_improvement = (students_improvement / total_students_naive * 100 if total_students_naive > 0 else 0)
courses_improvement = len(schedule) - len(naive_schedule)

print("\nComparison:")
print(f"ILP schedules {courses_improvement} more courses than naive solution")
print(f"ILP accommodates {students_improvement} more students ({percent_improvement:.1f}% improvement)")
wb_rooms = sum(1 for r in rooms if has_whiteboard[r])
pr_rooms = sum(1 for r in rooms if has_projector[r])
both_rooms = sum(1 for r in rooms if has_whiteboard[r] and has_projector[r])

wb_courses = sum(1 for c in courses if requires_whiteboard[c])
pr_courses = sum(1 for c in courses if requires_projector[c])
both_courses = sum(1 for c in courses if requires_whiteboard[c] and requires_projector[c])

wb_scheduled_ilp = sum(1 for c in schedule if requires_whiteboard[c])
pr_scheduled_ilp = sum(1 for c in schedule if requires_projector[c])
both_scheduled_ilp = sum(1 for c in schedule if requires_whiteboard[c] and requires_projector[c])

wb_scheduled_naive = sum(1 for c in naive_schedule if requires_whiteboard[c])
pr_scheduled_naive = sum(1 for c in naive_schedule if requires_projector[c])
both_scheduled_naive = sum(1 for c in naive_schedule if requires_whiteboard[c] and requires_projector[c])

print(f"\nResource statistics:")
print(f"Rooms: WB={wb_rooms}, PR={pr_rooms}, Both={both_rooms}")
print(f"Courses: WB={wb_courses}, PR={pr_courses}, Both={both_courses}")
print(
    f"ILP scheduled: WB={wb_scheduled_ilp}/{wb_courses}, PR={pr_scheduled_ilp}/{pr_courses}, Both={both_scheduled_ilp}/{both_courses}")
print(
    f"Naive scheduled: WB={wb_scheduled_naive}/{wb_courses}, PR={pr_scheduled_naive}/{pr_courses}, Both={both_scheduled_naive}/{both_courses}")