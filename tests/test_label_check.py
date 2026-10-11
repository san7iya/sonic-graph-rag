from src.evaluation.label_check import coarse_genre, compare_labels


def test_coarse_genre_maps_families_and_ignores_unhelpful_tags():
    assert coarse_genre("Hip-Hop/Rap") == "hiphop"
    assert coarse_genre("Reggae") == "reggae"
    assert coarse_genre("Vocal Jazz") == "jazz"
    assert coarse_genre("Classical") == "classical"
    assert coarse_genre("Hard Rock") == "rock"
    assert coarse_genre("Pop") == "pop" and coarse_genre("R&B/Soul") == "pop"
    assert coarse_genre("Soundtrack") is None and coarse_genre("") is None


def test_compare_labels_reports_agreement_and_disagreements():
    rows = [
        {"track_id": "a", "genre": "rock", "itunes_genre": "Rock"},
        {"track_id": "b", "genre": "rock", "itunes_genre": "Pop"},          # disagreement
        {"track_id": "c", "genre": "jazz", "itunes_genre": "Soundtrack"},    # not comparable
        {"track_id": "d", "genre": "pop", "itunes_genre": "Pop"},
    ]
    result = compare_labels(rows)
    assert result["mapped"] == 3 and abs(result["agreement"] - 2 / 3) < 1e-9
    assert result["per_genre"] == {"rock": (2, 1), "pop": (1, 1)}
    assert result["disagreements"] == [{"track_id": "b", "curated": "rock", "itunes": "Pop", "itunes_maps_to": "pop"}]
