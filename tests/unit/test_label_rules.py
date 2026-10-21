from scripts.label import parse_verdict, small_ok


def test_small_ok_rules():
    # Small is A in the first comparison and B in the second.
    assert small_ok("A", "B") == 1  # small wins both
    assert small_ok("B", "A") == 0  # small loses both
    assert small_ok("tie", "tie") == 1
    assert small_ok("B", "tie") == 0  # mean 0.25
    assert small_ok("A", "A") == 1  # one win, one loss: position bias, counts as a tie


def test_parse_verdict():
    assert parse_verdict("A") == "A"
    assert parse_verdict(" b.\n") == "B"
    assert parse_verdict("Tie") == "tie"
    assert parse_verdict("Answer A is better") is None
