import pytest

from app.schemas import ChatMessage


def test_content_parts_joined():
    content = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    assert ChatMessage(role="user", content=content).text() == "a\nb"


def test_image_part_rejected():
    content = [{"type": "image_url", "image_url": {"url": "http://x/y.png"}}]
    with pytest.raises(ValueError, match="unsupported_content"):
        ChatMessage(role="user", content=content).text()
