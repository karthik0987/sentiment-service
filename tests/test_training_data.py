from src import train


def test_load_data_creates_cache_directory_from_fresh_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        train,
        "load_dataset",
        lambda name, split: {
            "text": ["Great film", "Bad film", "Loved it", "Disliked it"],
            "label": [1, 0, 1, 0],
        },
    )

    assert not (tmp_path / "data").exists()
    loaded = train.load_data(n_samples=4)

    assert len(loaded) == 4
    assert (tmp_path / "data" / "imdb_sample.csv").is_file()
