import pytest

from pinocchio.utils.json_parser import parse_structured_output, safe_json_parse

# 1. Standard JSON
STANDARD_JSON = (
    '{"agent_type": "generator", "success": true, "output": {"code": "print(1)"}}'
)

# 2. JSON wrapped by extra text
WRAPPED_JSON = 'Here is your result: {"agent_type": "generator", "success": true, "output": {"code": "print(2)"}} Thank you!'

# 3. JSON wrapped in markdown code block
MARKDOWN_JSON = '```json\n{"agent_type": "generator", "success": true, "output": {"code": "print(3)"}}\n```'

# 4. Partial/truncated JSON
PARTIAL_JSON = '{"agent_type": "generator", "success": true, "output": {"code": "print(4)"}'  # missing closing brace

# 5. Completely unparsable text
PLAIN_TEXT = "This is not JSON at all."


@pytest.mark.parametrize(
    "input_str,expect_success,expect_code",
    [
        (STANDARD_JSON, True, "print(1)"),
        (WRAPPED_JSON, True, "print(2)"),
        (MARKDOWN_JSON, True, "print(3)"),
        (PARTIAL_JSON, True, "print(4)"),
        (PLAIN_TEXT, False, None),
    ],
)
def test_safe_json_parse_and_structured_output(input_str, expect_success, expect_code):
    # safe_json_parse
    parsed = safe_json_parse(input_str)
    if expect_success:
        assert parsed is not None
        assert parsed.get("agent_type") == "generator"
        assert parsed.get("success") is True
        assert "output" in parsed
        assert parsed["output"].get("code") == expect_code
    else:
        assert parsed is None or "content" in parsed

    # parse_structured_output
    structured = parse_structured_output(input_str)
    if expect_success:
        assert structured.get("agent_type") == "generator"
        assert structured.get("success") is True
        assert "output" in structured
        assert structured["output"].get("code") == expect_code
    else:
        assert structured.get("format") == "plain_text"
        assert structured.get("parsed") is False


# 6. Multiple JSON objects, should extract the first one
MULTI_JSON = '{"agent_type": "generator", "success": true, "output": {"code": "print(5)"}} some text {"agent_type": "debugger", "success": true}'


def test_multi_json_extract():
    parsed = safe_json_parse(MULTI_JSON)
    assert parsed is not None
    assert parsed.get("agent_type") == "generator"
    assert parsed.get("success") is True
    assert parsed["output"].get("code") == "print(5)"


# 7. Markdown code block + extra wrapping text
COMPLEX = 'Result: ```json\n{"agent_type": "generator", "success": true, "output": {"code": "print(6)"}}\n``` End.'


def test_markdown_and_wrap():
    structured = parse_structured_output(COMPLEX)
    assert structured.get("agent_type") == "generator"
    assert structured.get("success") is True
    assert structured["output"].get("code") == "print(6)"
