#!/usr/bin/env python3
"""Script to check for Chinese characters in files."""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Set


def contains_chinese_chars(text: str) -> bool:
    """
    Check if the given text contains Chinese characters.

    Args:
        text: The text to check

    Returns:
        True if the text contains Chinese characters, False otherwise
    """
    # Unicode ranges for Chinese characters
    # CJK Unified Ideographs: U+4E00 - U+9FFF
    # CJK Unified Ideographs Extension A: U+3400 - U+4DBF
    # CJK Unified Ideographs Extension B: U+20000 - U+2A6DF
    # CJK Unified Ideographs Extension C: U+2A700 - U+2B73F
    # CJK Unified Ideographs Extension D: U+2B740 - U+2B81F
    # CJK Unified Ideographs Extension E: U+2B820 - U+2CEAF
    # CJK Unified Ideographs Extension F: U+2CEB0 - U+2EBEF
    # CJK Compatibility Ideographs: U+F900 - U+FAFF
    # CJK Compatibility Ideographs Supplement: U+2F800 - U+2FA1F

    # Simplified regex pattern for Chinese characters
    pattern = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")
    return bool(pattern.search(text))


def is_in_string(line: str, char_pos: int) -> bool:
    """
    Check if a character position is inside a string literal.

    Args:
        line: The line of code
        char_pos: Position of the character to check

    Returns:
        True if the character is inside a string literal, False otherwise
    """
    # Simple heuristic to detect if we're inside a string
    # This handles basic cases but may not catch all edge cases

    # Count quotes before the position
    single_quotes = line[:char_pos].count("'")
    double_quotes = line[:char_pos].count('"')

    # Check for triple quotes
    triple_single = line[:char_pos].count("'''")
    triple_double = line[:char_pos].count('"""')

    # Adjust for triple quotes (each triple quote counts as 3 single/double quotes)
    single_quotes -= triple_single * 2  # Each triple quote uses 3 single quotes
    double_quotes -= triple_double * 2  # Each triple quote uses 3 double quotes

    # If we have an odd number of quotes, we're inside a string
    return (single_quotes % 2 == 1) or (double_quotes % 2 == 1)


def check_file(file_path: Path) -> List[int]:
    """
    Check a file for Chinese characters, ignoring those in strings.

    Args:
        file_path: Path to the file to check

    Returns:
        List of line numbers containing Chinese characters (outside strings)
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        # Skip binary files
        return []

    chinese_lines = []
    for i, line in enumerate(lines, 1):
        # Find all Chinese characters in the line
        pattern = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")
        matches = list(pattern.finditer(line))

        # Check if any Chinese characters are outside strings
        has_chinese_outside_string = False
        for match in matches:
            if not is_in_string(line, match.start()):
                has_chinese_outside_string = True
                break

        if has_chinese_outside_string:
            chinese_lines.append(i)

    return chinese_lines


def main() -> int:
    """
    Execute the Chinese character detection script.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(description="Check files for Chinese characters")
    parser.add_argument("files", nargs="+", help="Files to check")
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[".mdc", ".md"],
        help="File extensions to exclude (default: .mdc, .md)",
    )

    args = parser.parse_args()

    # Convert exclude list to set of extensions with dots
    exclude_exts: Set[str] = {
        ext if ext.startswith(".") else f".{ext}" for ext in args.exclude
    }

    has_chinese = False

    for file_name in args.files:
        file_path = Path(file_name)

        # Skip excluded file extensions
        if file_path.suffix in exclude_exts:
            continue

        chinese_lines = check_file(file_path)

        if chinese_lines:
            has_chinese = True
            line_list = ", ".join(map(str, chinese_lines))
            print(f"{file_path}: Chinese characters found on lines {line_list}")

    return 1 if has_chinese else 0


if __name__ == "__main__":
    sys.exit(main())
