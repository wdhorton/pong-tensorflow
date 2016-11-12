workspace(name = "pong")

local_repository(
    name = "tf_serving",
    path = "serving"
)

local_repository(
    name = "org_tensorflow",
    path = "serving/tensorflow",
)

load('//pong:workspace.bzl', 'pong_workspace')
pong_workspace()
