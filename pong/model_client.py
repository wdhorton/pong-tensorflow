from grpc.beta import implementations
import tensorflow as tf
import numpy as np

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

def make_prediction(data):
  channel = implementations.insecure_channel('localhost', 9000)
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'pong_model'
  request.inputs['data'].CopyFrom(
      tf.contrib.util.make_tensor_proto(data))
  result = stub.Predict(request, 1.0)
  return result

if __name__ == '__main__':
  json = {
    	"ball_y_velocity" : -5.900753227819585,
    	"paddle_position" : 187.5,
    	"ball_x_position" : 210.38522498396122,
    	"ball_y_position" : 76.78267576014937,
    	"ball_x_velocity" : -3.896294565914742,
  }
  new_sample = np.array([
    json["ball_x_velocity"],
    json["ball_y_velocity"],
    json["ball_x_position"],
    json["ball_y_position"],
    json["paddle_position"]
  ], dtype=np.float32)

  print make_prediction(new_sample)
