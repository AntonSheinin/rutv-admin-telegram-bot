from app.telegram.bot import extract_message, split_telegram_text


def test_split_telegram_text():
    chunks = split_telegram_text("x" * 8000)
    assert len(chunks) == 3
    assert all(len(chunk) <= 3900 for chunk in chunks)


def test_extract_message_returns_text_message():
    update = {
        "update_id": 10,
        "message": {
            "text": " hello ",
            "chat": {"id": 20},
            "from": {"id": 30},
        },
    }

    assert extract_message(update) == (10, 30, 20, "hello")


def test_extract_message_ignores_non_text_update():
    assert extract_message({"update_id": 10, "message": {"photo": []}}) is None
