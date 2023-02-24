import pandas as pd
import numpy as np
from falcon.type_guessing import determine_column_types
from falcon.types import ColumnTypes


def test_type_guessing_0():
    data = np.asarray([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]])

    type_ = determine_column_types(data)[0]

    assert type_ == ColumnTypes.NUMERIC_REGULAR


def test_type_guessing_1_from_num():
    data = np.asarray([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])

    type_ = determine_column_types(data)[0]

    assert type_ == ColumnTypes.CAT_LOW_CARD


def test_type_guessing_1():
    a = [[f"cat_{i}"] for i in range(40)]
    a.append(["cat_1"])
    data = np.asarray(a)

    type_ = determine_column_types(data)[0]

    assert type_ == ColumnTypes.CAT_LOW_CARD


def test_type_guessing_2():
    a = [[f"cat_{i}"] for i in range(400)]
    a.append(["cat_1"])
    data = np.asarray(a)

    type_ = determine_column_types(data)[0]

    assert type_ == ColumnTypes.CAT_HIGH_CARD


def test_type_guessing_3():
    corpus = [
        "Removed demands expense account in outward tedious do.",
        "Particular way thoroughly unaffected projection favourable mrs can projecting own.",
        "Thirty it matter enable become admire in giving."
        "Drawings offended yet answered jennings perceive laughing six did far.",
    ]

    type_ = determine_column_types(np.array(corpus))[0]

    assert type_ == ColumnTypes.TEXT_UTF8, 'corpus not detected'

    not_corpus = [
        "Removed demands expense",
        "Particular way thoroughly unaffected",
        "Thirty it matter enable become"
        "Drawings offended yet answered jennings.",
    ]

    type_ = determine_column_types(np.array(not_corpus))[0]

    assert type_ != ColumnTypes.TEXT_UTF8, 'not_corpus wrongly detected as text'

def test_type_guessing_100():
    data = np.asarray([["2022-02-02"], ["2022-02-25"], ["2022-05-02"]])

    type_ = determine_column_types(data)[0]

    assert type_ == ColumnTypes.DATE_YMD_ISO8601


def test_type_guessing_101():

    data = np.asarray(
        [["2022-02-02T12:13:14Z"], ["2022-02-25T15:16:17Z"], ["2022-05-02T18:19:20Z"]]
    )
    type_ = determine_column_types(data)[0]
    assert (
        type_ == ColumnTypes.DATETIME_YMDHMS_ISO8601
    ), r"%Y-%m-%dT%H:%M:%SZ format was not detected"

    data = np.asarray(
        [["2022-02-02 12:13:14"], ["2022-02-25 15:16:17"], ["2022-05-02 18:19:20"]]
    )
    type_ = determine_column_types(data)[0]
    assert (
        type_ == ColumnTypes.DATETIME_YMDHMS_ISO8601
    ), r"%Y-%m-%d %H:%M:%S format was not detected"
