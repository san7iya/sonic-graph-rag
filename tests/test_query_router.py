import numpy as np
import pytest

from src import config
from src.embeddings.fusion import apply_query_weights
from src.rag.query_router import EQUAL, route_query


def test_audio_cue_favours_audio():
    w = route_query("calm instrumental music with no vocals")
    assert w[2] == max(w) and sum(w) == pytest.approx(1)


def test_text_and_image_cues():
    assert route_query("songs about heartbreak")[0] == max(route_query("songs about heartbreak"))
    assert route_query("a colourful album cover")[1] == max(route_query("a colourful album cover"))


def test_no_cue_or_tie_keeps_equal_weights():
    assert route_query("something nice") == EQUAL
    assert route_query("slow songs about loss") == EQUAL     # one audio cue vs one text cue


def test_cue_must_be_a_whole_word():
    assert route_query("Rabout") == EQUAL                      # 'about' inside a word does not count


def test_equal_weights_are_identity_and_weights_scale_blocks():
    v = np.ones(config.FUSED_EMBED_DIM)
    assert np.allclose(apply_query_weights(v, EQUAL), v)
    scaled = apply_query_weights(v, (0.6, 0.2, 0.2))
    assert scaled[0] == pytest.approx(1.8) and scaled[config.TEXT_EMBED_DIM] == pytest.approx(0.6)
    assert v[0] == 1.0                                          # input not modified
