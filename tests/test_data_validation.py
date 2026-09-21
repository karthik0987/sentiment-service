import pandas as pd
import pytest

from src.train import validate_data


@pytest.fixture
def valid_data():
    return pd.DataFrame(
        {
            "text": ["Great film", "Loved it", "Bad film", "Disliked it"],
            "label": [1, 1, 0, 0],
        }
    )


def test_valid_data_passes(valid_data):
    validate_data(valid_data)


def test_missing_text_column_is_rejected(valid_data):
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_data(valid_data.drop(columns="text"))


def test_empty_dataset_is_rejected(valid_data):
    with pytest.raises(ValueError, match="dataset is empty"):
        validate_data(valid_data.iloc[0:0])


def test_missing_review_text_is_rejected(valid_data):
    valid_data.loc[0, "text"] = None
    with pytest.raises(ValueError, match="missing review text"):
        validate_data(valid_data)


def test_non_text_review_is_rejected(valid_data):
    valid_data["text"] = valid_data["text"].astype(object)
    valid_data.loc[0, "text"] = 123
    with pytest.raises(ValueError, match="non-text reviews"):
        validate_data(valid_data)


def test_whitespace_only_review_is_rejected(valid_data):
    valid_data.loc[0, "text"] = "   "
    with pytest.raises(ValueError, match="empty reviews"):
        validate_data(valid_data)


def test_missing_label_is_rejected(valid_data):
    valid_data.loc[0, "label"] = None
    with pytest.raises(ValueError, match="missing labels"):
        validate_data(valid_data)


def test_single_class_is_rejected(valid_data):
    valid_data["label"] = 1
    with pytest.raises(ValueError, match="Expected labels"):
        validate_data(valid_data)


def test_class_with_one_example_is_rejected(valid_data):
    valid_data.loc[2, "label"] = 1
    with pytest.raises(ValueError, match="at least two rows"):
        validate_data(valid_data)
