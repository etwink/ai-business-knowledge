"""Scan one or more directories and categorize files by type."""

from pathlib import Path
from dataclasses import dataclass, field


COBOL_EXTENSIONS = {".COB", ".CIC", ".CPY", ".MPS", ".SRC", ".CT1", ".JCV", ".PRV", ".CBL", ".COBOL"}

# Non-COBOL source code — analyzed as text, LLM-clustered (no dependency graph)
CODE_EXTENSIONS = {
    ".PY",   # Python
    ".SQL",  # SQL / T-SQL / PL-SQL / stored procedures
    ".JS",   # JavaScript
    ".TS",   # TypeScript
    ".VB",   # Visual Basic
    ".BAS",  # BASIC / VBA module exports
    ".CS",   # C#
    ".JAVA", # Java
    ".PS1",  # PowerShell
    ".R",    # R
    ".SH",   # Shell script
    ".BAT",  # Windows batch
    ".CMD",  # Windows command
}

WORD_EXTENSIONS = {".DOCX", ".DOC"}
EXCEL_EXTENSIONS = {".XLSX", ".XLS", ".XLSM", ".XLSB"}


@dataclass
class ScannedDocuments:
    cobol: list[Path] = field(default_factory=list)
    code: list[Path] = field(default_factory=list)   # Non-COBOL source code
    word: list[Path] = field(default_factory=list)
    excel: list[Path] = field(default_factory=list)

    @property
    def total_count(self) -> int:
        return len(self.cobol) + len(self.code) + len(self.word) + len(self.excel)

    def summary(self) -> str:
        parts = []
        if self.cobol:
            parts.append(f"{len(self.cobol)} COBOL")
        if self.code:
            parts.append(f"{len(self.code)} source code")
        if self.word:
            parts.append(f"{len(self.word)} Word")
        if self.excel:
            parts.append(f"{len(self.excel)} Excel")
        return ", ".join(parts) + f" ({self.total_count} total)"


class FolderScanner:
    """Scan one or more directories for supported file types."""

    def scan(self, paths: list[Path | str], recursive: bool = True) -> ScannedDocuments:
        result = ScannedDocuments()
        seen: set[Path] = set()

        for root in paths:
            root = Path(root)
            if not root.exists():
                continue
            pattern = "**/*" if recursive else "*"
            for p in sorted(root.glob(pattern)):
                if not p.is_file() or p in seen:
                    continue
                seen.add(p)
                ext = p.suffix.upper()
                if ext in COBOL_EXTENSIONS:
                    result.cobol.append(p)
                elif ext in CODE_EXTENSIONS:
                    result.code.append(p)
                elif ext in WORD_EXTENSIONS:
                    result.word.append(p)
                elif ext in EXCEL_EXTENSIONS:
                    result.excel.append(p)

        return result
