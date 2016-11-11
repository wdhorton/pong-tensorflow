py_binary(
    name = "server",
    srcs = [
        "server.py",
    ],
    deps = [
        ":model_client"
    ],
)

py_binary(
    name = "model_client",
    srcs = ["model_client.py"],
    deps = [
        "@tf_serving//tensorflow_serving/apis:predict_proto_py_pb2",
        "@tf_serving//tensorflow_serving/apis:prediction_service_proto_py_pb2",
    ]
)

local_repository(
    name = "tf_serving,
    path = "serving",
)
