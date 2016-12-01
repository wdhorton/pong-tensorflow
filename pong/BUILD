py_binary(
    name = "server",
    srcs = [
        "server.py",
    ],
    deps = [
        ":model_client",
        "//pong/training:load_data"
    ],
)

py_binary(
    name = "model_client",
    srcs = ["model_client.py"],
    deps = [
        "//pong/training:load_data",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@tf_serving//tensorflow_serving/apis:predict_proto_py_pb2",
        "@tf_serving//tensorflow_serving/apis:prediction_service_proto_py_pb2",
    ]
)
