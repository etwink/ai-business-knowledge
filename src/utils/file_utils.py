"""Utility functions for file handling and processing."""

from pathlib import Path
from typing import List, Optional
import config


def get_supported_documents(directory: Path) -> List[Path]:
    """Get all supported documents from a directory."""
    directory = Path(directory)
    if not directory.exists():
        return []

    documents = []
    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in config.SUPPORTED_FORMATS:
            documents.append(file_path)

    return sorted(documents)


def is_supported_format(file_path: Path) -> bool:
    """Check if file format is supported."""
    return Path(file_path).suffix.lower() in config.SUPPORTED_FORMATS


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in megabytes."""
    return Path(file_path).stat().st_size / (1024 * 1024)


def validate_file(file_path: Path, max_size_mb: Optional[int] = None) -> tuple[bool, Optional[str]]:
    """Validate a file for processing."""
    file_path = Path(file_path)

    if not file_path.exists():
        return False, f"File not found: {file_path}"

    if not is_supported_format(file_path):
        return False, f"Unsupported format: {file_path.suffix}"

    max_size = max_size_mb or config.MAX_DOCUMENT_SIZE_MB
    if get_file_size_mb(file_path) > max_size:
        return False, f"File exceeds maximum size of {max_size}MB"

    return True, None
