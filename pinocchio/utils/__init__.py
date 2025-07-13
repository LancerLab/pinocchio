"""Utilities for Pinocchio modules.

This package contains utilities for module functionality,
such as file operations, JSON parsing, temporary file management,
configuration helpers, string processing, validation, and other module-specific utilities.
"""

from .config_utils import (
    create_default_test_config,
    create_minimal_test_config,
    create_test_config,
    load_test_config,
    merge_configs,
    validate_config,
)
from .file_utils import (
    cleanup_old_files,
    ensure_directory,
    get_file_info,
    get_unique_filename,
    safe_read_json,
    safe_read_text,
    safe_write_json,
    safe_write_text,
)
from .json_parser import (
    extract_code_from_response,
    format_json_response,
    parse_structured_output,
    safe_json_parse,
    validate_agent_response,
)
from .string_utils import (
    extract_code_blocks,
    extract_urls,
    format_duration,
    format_file_size,
    is_valid_json,
    normalize_whitespace,
    remove_ansi_escape_codes,
    sanitize_filename,
    split_safe,
    truncate_text,
)
from .temp_utils import (
    cleanup_temp_directories,
    cleanup_temp_files,
    create_temp_directory,
    create_temp_file,
    get_temp_file_path,
)
from .validation import (
    assert_dict_structure,
    sanitize_input,
    validate_agent_response_format,
    validate_api_key,
    validate_config_schema,
    validate_email,
    validate_file_path,
    validate_json_structure,
    validate_memory_data,
    validate_session_data,
    validate_url,
)

__all__ = [
    # File utilities
    "ensure_directory",
    "safe_write_json",
    "safe_read_json",
    "safe_write_text",
    "safe_read_text",
    "get_unique_filename",
    "cleanup_old_files",
    "get_file_info",
    # JSON utilities
    "safe_json_parse",
    "parse_structured_output",
    "format_json_response",
    "validate_agent_response",
    "extract_code_from_response",
    # String utilities
    "sanitize_filename",
    "truncate_text",
    "extract_code_blocks",
    "normalize_whitespace",
    "split_safe",
    "format_duration",
    "format_file_size",
    "is_valid_json",
    "extract_urls",
    "remove_ansi_escape_codes",
    # Temporary file utilities
    "create_temp_file",
    "create_temp_directory",
    "cleanup_temp_files",
    "cleanup_temp_directories",
    "get_temp_file_path",
    # Configuration utilities
    "create_test_config",
    "load_test_config",
    "merge_configs",
    "create_default_test_config",
    "validate_config",
    "create_minimal_test_config",
    # Validation utilities
    "validate_file_path",
    "validate_json_structure",
    "validate_email",
    "validate_url",
    "validate_config_schema",
    "validate_agent_response_format",
    "validate_session_data",
    "validate_memory_data",
    "sanitize_input",
    "validate_api_key",
    "assert_dict_structure",
]
