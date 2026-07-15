"""MET-based calorie estimation.

Calories = MET x bodyweight(kg) x duration(hours).
MET values from the Compendium of Physical Activities (vigorous calisthenics
for bodyweight moves, moderate effort for squats).
"""

MET = {
    "push_up": 8.0,
    "pull_up": 8.0,
    "situp": 8.0,
    "squat": 5.0,
    "jumping_jack": 8.0,
}


def calories(exercise: str, weight_kg: float, duration_s: float) -> float:
    return MET[exercise] * weight_kg * (duration_s / 3600.0)
