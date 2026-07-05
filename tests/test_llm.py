from app.llm.openai import parse_openai_response


class Item:
    type = "function_call"
    call_id = "call_1"
    name = "refresh_playlist"
    arguments = '{"id":"x"}'


class Response:
    output_text = ""
    output = [Item()]


def test_parse_openai_tool_call():
    parsed = parse_openai_response(Response())
    assert parsed.tool_calls[0].name == "refresh_playlist"
    assert parsed.tool_calls[0].arguments == {"id": "x"}


def test_parse_openai_tool_call_malformed_arguments():
    class BadItem:
        type = "function_call"
        call_id = "call_1"
        name = "refresh_playlist"
        arguments = object()

    class BadResponse:
        output_text = ""
        output = [BadItem()]

    parsed = parse_openai_response(BadResponse())
    assert parsed.tool_calls[0].arguments == {}


def test_parse_openai_skips_malformed_tool_call_identity():
    class BadItem:
        type = "function_call"
        call_id = ""
        name = "refresh_playlist"
        arguments = "{}"

    class BadResponse:
        output_text = ""
        output = [BadItem()]

    parsed = parse_openai_response(BadResponse())
    assert parsed.tool_calls == []
