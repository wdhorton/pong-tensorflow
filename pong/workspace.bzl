load('@tf_serving//tensorflow-serving:workspace.bzl', 'tf_serving_workspace')

def pong_workspace():
  tf_serving_workspace()
