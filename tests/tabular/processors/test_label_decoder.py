from falcon.tabular.processors.label_decoder import LabelDecoder
import numpy as np


def test_label_decoder():
    labels = np.array(["A", "B", "C", "D"])
    processor = LabelDecoder()
    processor.fit(labels)
    expected_encoded_labels = np.array([0, 1, 2, 3])
    encoded_labels = processor.transform(labels, inverse=False)
    assert False not in np.equal(expected_encoded_labels, encoded_labels)
    decoed_labels = processor.transform(np.array([3, 2, 1, 0]), inverse=True)
    expected_decoed_labels = np.array(["D", "C", "B", "A"], dtype=np.str_)
    print(decoed_labels.dtype, expected_decoed_labels.dtype)
    assert False not in np.equal(
        decoed_labels.astype(np.object_), expected_decoed_labels.astype(np.object_)
    )
