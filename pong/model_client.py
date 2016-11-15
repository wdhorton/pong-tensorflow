from grpc.beta import implementations
import tensorflow as tf
import numpy as np

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

from load_data import scale_features

def make_prediction(data):
  channel = implementations.insecure_channel('localhost', 9000)
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'pong_model'
  request.inputs['data'].CopyFrom(
      tf.contrib.util.make_tensor_proto(data, shape=[1, data.size]))
  response = stub.Predict(request, 1.0)
  prediction = np.argmax(np.array(response.outputs['move'].float_val))
  return prediction

if __name__ == '__main__':
  from pymongo import MongoClient
  PONG_DB_NAME = 'pong'
  COLLECTION_NAME = 'game_data'
  game_data = MongoClient()[PONG_DB_NAME][COLLECTION_NAME]
  rows = game_data.find().limit(100)

  for row in rows:
    # Classes: UP -- 0, STATIONARY -- 1, DOWN -- 2
    if row['paddle_velocity'] < 0:
      target = np.array([1, 0, 0])
    elif row['paddle_velocity'] == 0:
      target = np.array([0, 1, 0])
    else:
      target = np.array([0, 0, 1])

    row_data = scale_features(row)
    new_sample = np.asarray(row_data, dtype=np.float32)

    print 'Predicted:'
    print make_prediction(new_sample)
    print 'Actual:'
    print np.argmax(target)
