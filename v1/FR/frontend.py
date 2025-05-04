
from flask import Flask, render_template, request

from dev.v0.DB.factories import RepositoryFactory
from dev.v0.WO import WorkoutFactory

import json

class AirFitApp:
    def __init__(self):
        self.app = Flask('AirFit')
        self.repo = RepositoryFactory.get_repository('sqlite')

        self.app.add_url_rule('/', 'create_workout', self.create_workout)
        self.app.add_url_rule('/submit_rating', 'save_workout', self.save_workout, methods=['POST'])
        self.app.add_url_rule('/api/workout', 'api_create_workout', self.api_create_workout)
        self.app.add_url_rule('/api/submit_rating', 'api_submit_rating', self.api_submit_rating, methods=['POST'])

        # GUI APP

    def api_submit_rating(self):
        result = json.loads(request.data)
        intensity = result['rating']
        workout_id = result['workout_id']
        self.repo.save_workout_intensity(workout_id, intensity)
        return f'{workout_id}, {intensity}'

    def general_workout(self):
        # self.repo.delete_unrated_workouts()  # Uncomment for testing!
        workout = self.repo.get_available_workout()
        if len(workout[0]) < 5:
            self.repo.delete_unrated_workouts()
            WorkoutFactory.workout_factory('random').generate()
            workout = self.repo.get_available_workout()
        workout_id = workout[0][0][0]
        workout = [(w[1], '-' if w[2] == 0 else w[2], w[3]) for w in workout[0]]
        return workout_id, workout

    def create_workout(self):
        workout_id, workout = self.general_workout()
        return render_template('workout.html', workout=workout, workout_id=workout_id)

    def api_create_workout(self):
        workout_id, workout = self.general_workout()
        # Already sorted by SQL, nothing more to be done
        return {'workout_id': workout_id, 'workout': workout}

    def save_workout(self):
        workout_id = request.form['workout_id']
        intensity = request.form['rating']
        self.repo.save_workout_intensity(workout_id, intensity)
        return f'{workout_id}, {intensity}'

    def run(self, host='0.0.0.0', debug=True):
        self.app.run(host=host, debug=debug)

if __name__ == '__main__':
    app = AirFitApp()
    app.run()