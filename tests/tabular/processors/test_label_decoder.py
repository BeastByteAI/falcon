import numpy as np

from falcon.tabular.processors.label_decoder import LabelDecoder
from falcon.types import ColumnTypes, DatasetSchema


def test_label_decoder() -> None:
    labels = np.array(["A", "B", "C", "D"])
    X = np.arange(4).reshape(-1, 1)
    schema = DatasetSchema(
        column_names=("feature",),
        column_types=(ColumnTypes.NUMERIC_REGULAR,),
        target_name="target",
        target_kind="classification",
        dimensions=X.shape,
    )
    processor = LabelDecoder()
    processor.fit(X, labels, schema)
    expected_encoded_labels = np.array([0, 1, 2, 3])
    encoded_labels = processor.encode(labels)
    assert False not in np.equal(expected_encoded_labels, encoded_labels)
    decoed_labels = processor.transform(np.array([3, 2, 1, 0]))
    expected_decoed_labels = np.array(["D", "C", "B", "A"], dtype=np.str_)
    print(decoed_labels.dtype, expected_decoed_labels.dtype)
    assert False not in np.equal(
        decoed_labels.astype(np.object_), expected_decoed_labels.astype(np.object_)
    )
