from flask import Flask, request, send_from_directory
from flask_socketio import SocketIO, emit
from pymongo import MongoClient
import numpy as np
import os

from model_client import make_prediction

PONG_DB_NAME = 'pong'
COLLECTION_NAME = 'game_data'

app = Flask(__name__)
socketio = SocketIO(app)

game_data = MongoClient()[PONG_DB_NAME][COLLECTION_NAME]

@app.route('/')
def root():
  return send_from_directory(os.path.join(os.getcwd(), 'pong'), 'index.html')

@app.route('/train')
def train():
  return send_from_directory(os.path.join(os.getcwd(), 'pong'), 'train.html')

@app.route('/play')
def play():
  return send_from_directory(os.path.join(os.getcwd(), 'pong'), 'play.html')

@app.route('/api/game_data', methods=['POST'])
def write_data():
  data = request.get_json()
  game_data.insert_many(data)
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

  prediction = make_prediction(new_sample)
  print prediction
  emit('move', { 'move': prediction })


if __name__ == "__main__":
    socketio.run(app)
