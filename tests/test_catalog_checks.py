from src.ingestion.artist_match import primary_artist, same_primary_artist
from src.preprocessing.catalog_checks import catalog_issues


def test_same_primary_artist_rejects_lookalikes_and_features_but_accepts_groups():
    assert not same_primary_artist("Nas", "Lil Nas X")
    assert not same_primary_artist("50 Cent", "Jeremih feat. 50 Cent")
    assert same_primary_artist("Bob Marley & The Wailers", "Bob Marley & The Wailers")
    assert same_primary_artist("Toots and the Maytals", "Toots & The Maytals")
    assert same_primary_artist("Jay-Z", "JAY-Z & Kanye West")
    assert same_primary_artist("The Notorious B.I.G.", "The Notorious B.I.G.")


def test_trailing_x_in_a_name_is_not_a_separator():
    assert primary_artist("Lil Nas X") == "lil nas x"


def test_catalog_issues_flags_wrong_artist_and_nonclassical_classical():
    row = {"artist": "Nas", "genre": "hiphop"}
    bad = catalog_issues(row, {"artist_ok": "no", "itunes_artist": "Lil Nas X", "itunes_genre": "Hip-Hop/Rap"})
    assert len(bad) == 1 and "Lil Nas X" in bad[0]

    classical = {"artist": "Mozart", "genre": "classical"}
    assert catalog_issues(classical, {"artist_ok": "na", "itunes_genre": "Classical"}) == []
    assert len(catalog_issues(classical, {"artist_ok": "na", "itunes_genre": "Hip-Hop/Rap"})) == 1
    assert catalog_issues(classical, {"artist_ok": "na", "itunes_genre": "Soundtrack"}) == []   # no evidence either way


def test_no_enrichment_means_no_checks():
    assert catalog_issues({"artist": "A", "genre": "rock"}, None) == []
