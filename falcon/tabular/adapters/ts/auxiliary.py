import pandas as pd
import numpy as np
from typing import Tuple, List
from onnx import ModelProto, TensorProto, helper as h, NodeProto


def _create_window(df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    for i in range(window_size):
        col_name = f"X{window_size - i}"
        df[col_name] = df["y"].shift(i + 1)
    df = df.iloc[window_size:]
    return df[df.columns[::-1]]


def _split_fn(
    X: np.ndarray, y: np.ndarray, eval_size: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_ind = int(len(X) * (1 - eval_size))
    X_train, X_test = X[:train_ind], X[train_ind:]
    y_train, y_test = y[:train_ind], y[train_ind:]
    return X_train, X_test, y_train, y_test


def _wrap_onnx(m: ModelProto) -> None:
    input_names_inorder = [i.name for i in m.graph.input]
    new_names = [f"new_input_{i}" for i in range(len(input_names_inorder))]

    for i, _ in reversed(list(enumerate(m.graph.input))):
        del m.graph.input[i]

    additional_nodes: List[NodeProto] = []

    inputs = [
        h.make_tensor_value_info(new_names[i], TensorProto.FLOAT, [None, 1])
        for i in range(len(input_names_inorder))
    ]

    sample_mean = h.make_node(
        "Mean",
        inputs=new_names,
        outputs=["sample_mean"],
        domain="",
        name="calculate_sample_mean",
    )
    additional_nodes.insert(0, sample_mean)
    reshape_shape = h.make_tensor(
        name="shape_",
        data_type=TensorProto.INT64,
        dims=[2],
        vals=np.asarray([-1, 1]).astype(np.int64),
    )
    m.graph.initializer.append(reshape_shape)
    reshape_mean = h.make_node(
        "Reshape",
        inputs=["sample_mean", "shape_"],
        outputs=["sample_mean_reshaped"],
        domain="",
        name="reshape_mean",
    )
    additional_nodes.insert(0, reshape_mean)
    array_feature_extract_out_names = []
    for i in range(len(new_names)):
        node_out_name = f"sample_normalized_{i}"
        sub = h.make_node(
            "Sub",
            inputs=[new_names[i], "sample_mean_reshaped"],
            outputs=[node_out_name],
            domain="",
            name=f"normalize_sample_{i}",
        )
        additional_nodes.insert(0, sub)
        array_feature_extract_out_names.append(node_out_name)

    reshape_output = h.make_node(
        "Reshape",
        inputs=["model_result", "shape_"],
        outputs=["model_result_reshaped"],
        domain="",
        name="reshape_model_result",
    )
    additional_nodes.insert(0, reshape_output)
    add_ = h.make_node(
        "Add",
        inputs=["model_result_reshaped", "sample_mean_reshaped"],
        outputs=["result_denormalized"],
        domain="",
        name="add_mean",
    )
    additional_nodes.insert(0, add_)
    squeeze = h.make_node(
        "Squeeze",
        inputs=["result_denormalized"],
        outputs=["ts_output"],
        domain="",
        name="squeeze_result",
    )
    additional_nodes.insert(0, squeeze)

    for n in inputs:
        m.graph.input.append(n)

    for i in additional_nodes:
        m.graph.node.insert(0, i)

    new_output = h.make_tensor_value_info("ts_output", TensorProto.FLOAT, [None, 1])
    old_output_name = m.graph.output[0].name
    del m.graph.output[0]
    m.graph.output.append(new_output)

    for n in m.graph.node:
        for (old_n, new_n) in zip(input_names_inorder, array_feature_extract_out_names):
            for i, inp in enumerate(n.input):
                if inp == old_n:
                    n.input[i] = new_n
            for i, out in enumerate(n.output):
                if out == old_output_name:
                    n.output[i] = "model_result"
