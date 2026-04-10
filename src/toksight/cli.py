"""CLI for toksight."""

from __future__ import annotations

import json
import sys
from typing import Optional


def _build_cli():  # type: ignore[no-untyped-def]
    try:
        import click
    except ImportError:
        raise SystemExit("CLI dependencies required: pip install toksight[cli]")

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()

    @click.group()
    @click.version_option(package_name="toksight")
    def cli() -> None:
        """toksight — tokenizer analysis toolkit."""

    @cli.command()
    @click.argument("tokenizer_name")
    @click.option("--json-out", "-o", type=click.Path(), default=None)
    def info(tokenizer_name: str, json_out: Optional[str]) -> None:
        """Show vocabulary statistics for a tokenizer."""
        from toksight.loader import load_tiktoken
        from toksight.stats import vocab_stats

        tok = load_tiktoken(tokenizer_name)
        stats = vocab_stats(tok)

        table = Table(title=f"Vocabulary Stats: {tokenizer_name}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Vocab size", f"{stats.vocab_size:,}")
        table.add_row("Special tokens", str(stats.n_special_tokens))
        table.add_row("Single-byte tokens", str(stats.n_byte_tokens))
        table.add_row("Single-char tokens", str(stats.n_single_char))
        table.add_row("Multi-word tokens", str(stats.n_multiword))
        table.add_row("Avg token length", f"{stats.avg_token_length:.2f} bytes")
        table.add_row("Max token length", f"{stats.max_token_length} bytes")

        console.print(table)

        if stats.script_coverage:
            script_table = Table(title="Top Scripts in Vocabulary")
            script_table.add_column("Script", style="cyan")
            script_table.add_column("Count", justify="right")
            for script, count in list(stats.script_coverage.items())[:15]:
                script_table.add_row(script, str(count))
            console.print(script_table)

        if json_out:
            import dataclasses
            data = dataclasses.asdict(stats)
            with open(json_out, "w") as f:
                json.dump(data, f, indent=2)
            console.print(f"[dim]Saved to {json_out}[/dim]")

    @cli.command()
    @click.argument("tokenizer_name")
    @click.option("--corpus", "-c", type=click.Path(exists=True), help="Text file corpus.")
    @click.option("--text", "-t", type=str, help="Inline text to analyze.")
    def compress(tokenizer_name: str, corpus: Optional[str], text: Optional[str]) -> None:
        """Measure compression ratio on text."""
        from toksight.compression import compute_compression
        from toksight.loader import load_tiktoken

        tok = load_tiktoken(tokenizer_name)
        texts = []
        if corpus:
            with open(corpus) as f:
                texts = [line.strip() for line in f if line.strip()]
        elif text:
            texts = [text]
        else:
            console.print("[red]Provide --corpus or --text[/red]", err=True)
            sys.exit(1)

        stats = compute_compression(tok, texts)
        table = Table(title=f"Compression: {tokenizer_name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_row("Total chars", f"{stats.total_chars:,}")
        table.add_row("Total bytes", f"{stats.total_bytes:,}")
        table.add_row("Total tokens", f"{stats.total_tokens:,}")
        table.add_row("Bytes/token", f"{stats.bytes_per_token:.2f}")
        table.add_row("Chars/token", f"{stats.chars_per_token:.2f}")
        table.add_row("Fertility (tokens/word)", f"{stats.fertility:.2f}")
        console.print(table)

    @cli.command()
    @click.argument("tokenizer_name")
    @click.option("--blocks", "-b", multiple=True, help="Unicode blocks to test.")
    @click.option("--json-out", "-o", type=click.Path(), default=None)
    def coverage(tokenizer_name: str, blocks: tuple, json_out: Optional[str]) -> None:
        """Analyze Unicode coverage."""
        from toksight.coverage import analyze_coverage
        from toksight.loader import load_tiktoken

        tok = load_tiktoken(tokenizer_name)
        block_list = list(blocks) if blocks else None
        result = analyze_coverage(tok, blocks=block_list)

        table = Table(title=f"Unicode Coverage: {tokenizer_name}")
        table.add_column("Block", style="cyan")
        table.add_column("Tested", justify="right")
        table.add_column("Covered", justify="right")
        table.add_column("Ratio", justify="right")

        for block_name, info in result.blocks_analyzed.items():
            ratio = info["ratio"]
            style = "green" if ratio > 0.95 else "yellow" if ratio > 0.5 else "red"
            table.add_row(
                block_name,
                str(info["tested"]),
                str(info["covered"]),
                Text(f"{ratio:.1%}", style=style),
            )

        table.add_section()
        overall_style = "green" if result.coverage_ratio > 0.95 else "yellow"
        table.add_row(
            "TOTAL",
            str(result.total_codepoints_tested),
            str(result.codepoints_covered),
            Text(f"{result.coverage_ratio:.1%}", style=overall_style),
        )
        console.print(table)

        if json_out:
            import dataclasses
            with open(json_out, "w") as f:
                json.dump(dataclasses.asdict(result), f, indent=2)

    @cli.command()
    @click.argument("tokenizer_name")
    @click.option("--max-tokens", "-n", type=int, default=None, help="Max tokens to audit.")
    def audit(tokenizer_name: str, max_tokens: Optional[int]) -> None:
        """Audit a tokenizer for glitch tokens and oddities."""
        from toksight.audit import audit as run_audit
        from toksight.loader import load_tiktoken

        tok = load_tiktoken(tokenizer_name)
        result = run_audit(tok, max_tokens=max_tokens)

        if not result.findings:
            console.print("[green]No issues found.[/green]")
            return

        table = Table(title=f"Audit: {tokenizer_name}")
        table.add_column("Category", style="cyan")
        table.add_column("Severity")
        table.add_column("Token ID", justify="right", style="dim")
        table.add_column("Token", max_width=30)
        table.add_column("Description", max_width=50)

        sev_styles = {"critical": "red bold", "warning": "yellow", "info": "blue"}
        for f in result.findings[:50]:
            table.add_row(
                f.category,
                Text(f.severity.upper(), style=sev_styles.get(f.severity, "")),
                str(f.token_id),
                f.token_text[:30],
                f.description[:50],
            )

        console.print(table)
        console.print(
            f"\n[dim]{len(result.findings)} findings total "
            f"({result.n_critical} critical, {result.n_warnings} warnings)[/dim]"
        )

    return cli


cli = _build_cli()

if __name__ == "__main__":
    cli()
