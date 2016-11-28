from flask import Flask, request, send_from_directory
from flask_socketio import SocketIO, emit
from pymongo import MongoClient
import numpy as np
import os

from model_client import make_prediction
from load_data import scale_features

PONG_DB_NAME = 'pong'
COLLECTION_NAME = 'game_data'
BINARY_COLLECTION_NAME = 'game_data_binary'

app = Flask(__name__)
socketio = SocketIO(app)

game_data = MongoClient()[PONG_DB_NAME][COLLECTION_NAME]
game_data_binary = MongoClient()[PONG_DB_NAME][BINARY_COLLECTION_NAME]

# @app.route('/')
# def root():
#   return send_from_directory(os.path.join(os.getcwd(), 'pong'), 'index.html')

@app.route('/train')
def train():
  return send_from_directory(os.path.join(os.getcwd(), 'pong'), 'train.html')

@app.route('/train_binary')
def train_binary():
  return send_from_directory(os.path.join(os.getcwd(), 'pong'), 'train_binary.html')

@app.route('/play')
def play():
  return send_from_directory(os.path.join(os.getcwd(), 'pong'), 'play.html')

@app.route('/api/game_data', methods=['POST'])
def write_data():
  data = request.get_json()
  game_data.insert_many(data)
  return 'ok'

@app.route('/api/game_data_binary', methods=['POST'])
def write_data_binary():
  data = request.get_json()
  game_data_binary.insert_many(data)
  return 'ok'

@socketio.on('current data')
def handle_new_data(json):
  new_sample = np.array(scale_features(json), dtype=np.float32)

  prediction = make_prediction(new_sample)
  print prediction
  emit('move', { 'move': prediction })


if __name__ == "__main__":
    socketio.run(app)
