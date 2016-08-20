from flask import Flask, request
from pymongo import MongoClient

PONG_DB_NAME = 'pong'
COLLECTION_NAME = 'game_data'

app = Flask(__name__, static_folder='')

game_data = MongoClient()[PONG_DB_NAME][COLLECTION_NAME]

@app.route('/')
def root():
  return app.send_static_file('index.html')

@app.route('/api/game_data', methods=['POST'])
def write_data():
  data = request.get_json()
  game_data.insert_many(data)
  return 'ok'

if __name__ == "__main__":
    app.run()
