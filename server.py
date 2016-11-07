from flask import Flask, request
from flask_socketio import SocketIO, emit
from pymongo import MongoClient
import numpy as np

from train_model import train_model

PONG_DB_NAME = 'pong'
COLLECTION_NAME = 'game_data'

app = Flask(__name__, static_folder='')
socketio = SocketIO(app)

game_data = MongoClient()[PONG_DB_NAME][COLLECTION_NAME]
classifier = train_model()

@app.route('/')
def root():
  return app.send_static_file('index.html')

@app.route('/api/game_data', methods=['POST'])
def write_data():
  data = request.get_json()
  # game_data.insert_many(data)
  return 'ok'

@socketio.on('current data')
def handle_new_data(json):
  new_sample = np.array([
    json["ball_x_velocity"],
    json["ball_y_velocity"],
    json["ball_x_position"],
    json["ball_y_position"],
    json["paddle_position"]
  ], dtype=np.float32)

  prediction = classifier.predict(new_sample)
  print prediction
  emit('move', { 'move': prediction[0] })


if __name__ == "__main__":
    socketio.run(app)
