from src.ingestion.artist_match import credited_artists, is_credited_artist
from src.preprocessing.catalog_checks import catalog_issues


def test_credited_artist_rejects_lookalikes_but_accepts_collaborations():
    assert not is_credited_artist("Nas", "Lil Nas X")
    assert not is_credited_artist("Culture", "Culture Club")
    assert is_credited_artist("50 Cent", "Jeremih & 50 Cent")              # credited collaborator
    assert is_credited_artist("Bill Evans", "Stan Getz & Bill Evans")
    assert is_credited_artist("Bob Marley & The Wailers", "Bob Marley & The Wailers")
    assert is_credited_artist("Toots and the Maytals", "Toots & The Maytals")
    assert is_credited_artist("Jay-Z", "JAY-Z & Kanye West")
    assert is_credited_artist("The Notorious B.I.G.", "The Notorious B.I.G.")


def test_credit_splitting_keeps_trailing_x_and_splits_features():
    assert credited_artists("Lil Nas X") == ["lil nas x"]
    assert credited_artists("Jeremih feat. 50 Cent") == ["jeremih", "50 cent"]
    assert credited_artists("Bugle, Damian Marley & Buju Banton") == ["bugle", "damian marley", "buju banton"]


def test_catalog_issues_flags_wrong_artist_and_nonclassical_classical():
    row = {"artist": "Nas", "genre": "hiphop"}
    bad = catalog_issues(row, {"artist_ok": "no", "itunes_artist": "Lil Nas X", "itunes_genre": "Hip-Hop/Rap"})
    assert len(bad) == 1 and "Lil Nas X" in bad[0]

    classical = {"artist": "Mozart", "genre": "classical"}
    assert catalog_issues(classical, {"artist_ok": "na", "itunes_genre": "Classical"}) == []
    assert len(catalog_issues(classical, {"artist_ok": "na", "itunes_genre": "Hip-Hop/Rap"})) == 1
    assert catalog_issues(classical, {"artist_ok": "na", "itunes_genre": "Soundtrack"}) == []   # no evidence either way
    assert catalog_issues(classical, {"artist_ok": "na", "itunes_genre": "Alternative"}) == []   # real Satie is tagged so


def test_no_enrichment_means_no_checks():
    assert catalog_issues({"artist": "A", "genre": "rock"}, None) == []
