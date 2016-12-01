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
    name="train_model_binary",
    srcs = ["train_model_binary.py"],
    deps = [
        ":load_data",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/contrib/session_bundle:exporter"
    ]
)

py_binary(
    name="train_model_hidden",
    srcs = ["train_model_hidden.py"],
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
        "@org_tensorflow//tensorflow/contrib/session_bundle:exporter"
    ]
)

py_binary(
    name="load_data",
    srcs=["load_data.py"],
    visibility=["//visibility:public"]
)
