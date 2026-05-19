"""Base document loader class."""

from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass


@dataclass
class DocumentContent:
    """Represents loaded document content."""
    filename: str
    file_type: str
    content: str
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseDocumentLoader(ABC):
    """Abstract base class for document loaders."""

    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    @abstractmethod
    def load(self) -> DocumentContent:
        """Load document content. Must be implemented by subclasses."""
        pass

    def _validate_file_size(self, max_size_mb: int = 50) -> bool:
        """Validate file size doesn't exceed maximum."""
        size_mb = self.file_path.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            raise ValueError(f"File size {size_mb:.2f}MB exceeds limit of {max_size_mb}MB")
        return True
