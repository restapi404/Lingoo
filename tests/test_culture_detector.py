import pytest
from culture_detector import detect_culture


def test_detect_country():
    assert detect_culture("I love India and its food.")[0] == "India"
    assert detect_culture("Welcome to Brazil!")[0] == "Brazil"


def test_detect_state():
    assert detect_culture("I visited Kerala last year")[0] == "Kerala"


def test_no_match():
    assert detect_culture("Random text")[0] is None
