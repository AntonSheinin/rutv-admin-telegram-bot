from app.telegram_bot import split_telegram_text


def test_split_telegram_text():
    chunks = split_telegram_text("x" * 8000)
    assert len(chunks) == 3
    assert all(len(chunk) <= 3900 for chunk in chunks)

