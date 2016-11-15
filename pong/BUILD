py_binary(
    name = "server",
    srcs = [
        "server.py",
    ],
    deps = [
        ":model_client",
        ":load_data"
    ],
)

py_binary(
    name = "model_client",
    srcs = ["model_client.py"],
    deps = [
        ":load_data",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@tf_serving//tensorflow_serving/apis:predict_proto_py_pb2",
        "@tf_serving//tensorflow_serving/apis:prediction_service_proto_py_pb2",
    ]
)

py_binary(
    name="train_model_2",
    srcs = ["train_model_2.py"],
    deps = [
        ":load_data",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/contrib/session_bundle:exporter"
    ]
)

py_binary(
    name="train_model",
    srcs = ["train_model.py"],
    deps = [
        ":load_data",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ]
)

py_binary(
    name="load_data",
    srcs=["load_data.py"]
)
