"""Load grants from the xlsx file and turn each row into a structured chunk
suitable for embedding. Every chunk is fully self-contained — the LLM never
needs to see the raw spreadsheet, only the rendered chunks we pass back.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import openpyxl


# Canonical column order in the spreadsheet. We pin these so a reordered
# spreadsheet doesn't silently corrupt the index.
EXPECTED_HEADERS: list[str] = [
    "Название программы",
    "Тип",
    "Организация",
    "Сфера",
    "Страна",
    "Дедлайн",
    "Сумма",
    "Доступность",
    "Место реализации",
    "Ключевые требования",
    "Сайт",
    "Статус",
]


@dataclass
class Grant:
    row_id: int
    name: str
    type: str
    organization: str
    field: str
    country: str
    deadline: str
    amount: str
    availability: str
    location: str
    requirements: str
    website: str
    status: str

    def to_chunk_text(self) -> str:
        """Render the grant as labeled key:value lines.

        Labels stay in Russian (matching the source) so the LLM grounds in
        the same vocabulary the user is likely to ask in. Empty fields are
        dropped so the embedding focuses on real signal.
        """
        lines = [f"Грант №{self.row_id}: {self.name}"]
        fields = [
            ("Тип", self.type),
            ("Организация", self.organization),
            ("Сфера", self.field),
            ("Страна", self.country),
            ("Дедлайн", self.deadline),
            ("Сумма", self.amount),
            ("Доступность", self.availability),
            ("Место реализации", self.location),
            ("Ключевые требования", self.requirements),
            ("Сайт", self.website),
            ("Статус", self.status),
        ]
        for label, value in fields:
            if value and value.strip() and value.strip().lower() not in {"none", "nan"}:
                lines.append(f"{label}: {value}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cell_to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, _dt.datetime):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, _dt.date):
        return value.strftime("%Y-%m-%d")
    return str(value).strip()


def load_grants(xlsx_path: str | Path) -> list[Grant]:
    """Read the xlsx and return a list of Grant records.

    Raises if the header row doesn't match EXPECTED_HEADERS — better to fail
    loudly than silently embed the wrong column.
    """
    path = Path(xlsx_path)
    if not path.exists():
        raise FileNotFoundError(f"Grants xlsx not found at: {path}")

    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.active

    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        raise ValueError("Spreadsheet is empty")

    headers = [_cell_to_str(c) for c in rows[0]]
    if headers[: len(EXPECTED_HEADERS)] != EXPECTED_HEADERS:
        raise ValueError(
            "Header row does not match expected schema.\n"
            f"  expected: {EXPECTED_HEADERS}\n"
            f"  got:      {headers}"
        )

    grants: list[Grant] = []
    for idx, row in enumerate(rows[1:], start=1):
        # Skip fully empty rows so trailing blanks in the sheet don't pollute
        # the index.
        cells = [_cell_to_str(c) for c in row]
        if not any(cells):
            continue
        # Pad short rows just in case
        while len(cells) < len(EXPECTED_HEADERS):
            cells.append("")
        grants.append(
            Grant(
                row_id=idx,
                name=cells[0],
                type=cells[1],
                organization=cells[2],
                field=cells[3],
                country=cells[4],
                deadline=cells[5],
                amount=cells[6],
                availability=cells[7],
                location=cells[8],
                requirements=cells[9],
                website=cells[10],
                status=cells[11],
            )
        )

    return grants


if __name__ == "__main__":
    # Smoke test — run `python -m bot.data` from the project root.
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "data/grants.xlsx"
    grants = load_grants(src)
    print(f"Loaded {len(grants)} grants from {src}\n")
    print("=== First grant chunk ===")
    print(grants[0].to_chunk_text())
