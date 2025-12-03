from flask import Flask, render_template
import pandas as pd

from v2.Workout.WorkoutGenerator import WorkoutGenerator

app = Flask(__name__)

def df_to_blocks(df: pd.DataFrame) -> list[dict]:

    blocks: list[dict] = []
    df = df.copy()

    for core_value, block_df in df.groupby("block"):
        block_df = block_df.sort_values("seq")

        exercises = []
        for _, row in block_df.iterrows():
            exercises.append(
                {
                    "seq": int(row["seq"]),
                    "name": str(row["exercise"]),
                    "weight": int(row["weight"]) if row["weight"] else 0,
                    "reps": int(row["reps"]),
                    "equipment": str(row["equipment"]),
                }
            )

        if core_value == 0:
            subtitle = "Block A"
        else:
            subtitle = "Block B"

        blocks.append(
            {
                "subtitle": subtitle,
                "exercises": exercises,
            }
        )
    return blocks


@app.route("/")
def show_workout():
    wg = WorkoutGenerator()
    df = wg.get_clean_workout()
    blocks = df_to_blocks(df)
    return render_template("workout.html", blocks=blocks)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')