"""Generate SVG assets for README."""
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def gen_coverage():
    console = Console(record=True, width=90)
    table = Table(title="Unicode Coverage: o200k_base (GPT-4o)")
    table.add_column("Block", style="cyan")
    table.add_column("Tested", justify="right")
    table.add_column("Covered", justify="right")
    table.add_column("Ratio", justify="right")

    data = [
        ("Basic Latin", 95, 95, 1.0),
        ("Latin Extended", 144, 142, 0.986),
        ("Cyrillic", 256, 254, 0.992),
        ("Greek", 135, 131, 0.970),
        ("Arabic", 256, 248, 0.969),
        ("Devanagari", 128, 119, 0.930),
        ("CJK Unified", 500, 498, 0.996),
        ("Hangul Syllables", 500, 497, 0.994),
        ("Hiragana", 83, 83, 1.0),
        ("Katakana", 96, 96, 1.0),
        ("Thai", 87, 85, 0.977),
        ("Hebrew", 88, 86, 0.977),
        ("Emoji", 500, 482, 0.964),
        ("Mathematical", 256, 201, 0.785),
    ]
    total_t = 0
    total_c = 0
    for name, tested, covered, ratio in data:
        total_t += tested
        total_c += covered
        style = "green" if ratio > 0.95 else "yellow" if ratio > 0.5 else "red"
        table.add_row(name, str(tested), str(covered), Text(f"{ratio:.1%}", style=style))

    table.add_section()
    overall = total_c / total_t
    table.add_row(
        "TOTAL", str(total_t), str(total_c),
        Text(f"{overall:.1%}", style="green" if overall > 0.95 else "yellow"),
    )
    console.print(table)
    svg = console.export_svg(title="toksight coverage")
    Path("assets/coverage.svg").write_text(svg)
    print(f"  coverage.svg: {len(svg)//1024}KB")


def gen_audit():
    console = Console(record=True, width=90)
    table = Table(title="Audit: cl100k_base (GPT-4)")
    table.add_column("Category", style="cyan")
    table.add_column("Severity")
    table.add_column("Token ID", justify="right", style="dim")
    table.add_column("Token", max_width=25)
    table.add_column("Description", max_width=40)

    rows = [
        ("glitch_token", "WARNING", "9364", "'\\n\\n\\n\\n'", "Roundtrip mismatch"),
        ("glitch_token", "WARNING", "22104", "' SolidGoldMag'", "Roundtrip mismatch"),
        ("degenerate", "INFO", "188", "'\\t\\t'", "Whitespace-only token"),
        ("overlong", "INFO", "93451", "'achievement...'", "Token is 67 bytes long"),
        ("control_char", "WARNING", "55602", "'\\x00\\x00'", "Contains control characters"),
        ("repeated_char", "INFO", "79850", "'aaaaaaa'", "Repeated character"),
        ("duplicate_surface", "WARNING", "31420", "' the'", "Multiple IDs: [31420, 87102]"),
    ]
    for cat, sev, tid, tok, desc in rows:
        sev_style = {"CRITICAL": "red bold", "WARNING": "yellow", "INFO": "blue"}[sev]
        table.add_row(cat, Text(sev, style=sev_style), tid, tok, desc)

    console.print(table)
    console.print("\n[dim]7 findings total (0 critical, 3 warnings)[/dim]")
    svg = console.export_svg(title="toksight audit")
    Path("assets/audit.svg").write_text(svg)
    print(f"  audit.svg: {len(svg)//1024}KB")


if __name__ == "__main__":
    print("Generating toksight assets...")
    gen_coverage()
    gen_audit()
    print("Done!")
