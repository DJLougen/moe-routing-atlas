"""MoE Atlas CLI — command-line interface for the routing atlas toolkit."""

import webbrowser
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .__version__ import __version__
from .config import get_config

console = Console()


def _banner():
    """Print the MoE Atlas banner."""
    console.print(
        Panel.fit(
            "[bold cyan]MoE Routing Atlas[/bold cyan]\n"
            f"[dim]v{__version__} — Map Mixture-of-Experts routing patterns[/dim]",
            border_style="cyan",
        )
    )


@click.group()
@click.version_option(version=__version__, prog_name="moe-atlas")
def cli():
    """MoE Routing Atlas — visualize and share expert routing patterns."""
    pass


@cli.command()
@click.option("--host", default=None, help="Server host (default: 127.0.0.1)")
@click.option("--port", default=None, type=int, help="Server port (default: 8000)")
@click.option("--db", default=None, help="SQLite database path")
@click.option("--reload", is_flag=True, help="Enable auto-reload (dev only)")
def serve(host, port, db, reload):
    """Start the visualization backend server."""
    _banner()
    config = get_config()

    host = host or config.backend_host
    port = port or config.backend_port
    db_path = db or str(config.db_path)

    console.print(f"[green]Starting backend server on http://{host}:{port}[/green]")
    console.print(f"[dim]Database: {db_path}[/dim]\n")

    from .backend import create_app
    import uvicorn

    app = create_app(db_path=db_path)
    uvicorn.run(app, host=host, port=port, reload=reload)


@cli.command()
@click.argument("text", required=False)
@click.option("--model", "-m", default=None, help="HuggingFace model ID")
@click.option("--quant", "-q", default=None, help="Quantization (nb4, nf4, int8, none)")
@click.option("--device", "-d", default=None, help="Device (cuda, mps, cpu)")
@click.option("--backend", "-b", default=None, help="Backend URL to send traces to")
@click.option("--file", "-f", type=click.Path(exists=True), help="Read text from file")
@click.option("--output", "-o", type=click.Path(), help="Save trace to file instead of sending")
@click.option(
    "--trust-remote-code",
    is_flag=True,
    default=None,
    help="Allow HuggingFace trust_remote_code (required for some MoE models)",
)
def trace(text, model, quant, device, backend, file, output, trust_remote_code):
    """Trace expert routing for a given text through an MoE model."""
    _banner()
    config = get_config()

    if file:
        text = Path(file).read_text(encoding="utf-8").strip()
    if not text:
        text = click.prompt("Enter text to trace")

    model = model or config.default_model
    quant = quant or config.default_quantization
    device = device or config.default_device
    backend_url = backend or config.backend_url

    console.print(f"[cyan]Model:[/cyan] {model}")
    console.print(f"[cyan]Quant:[/cyan] {quant}")
    console.print(f"[cyan]Device:[/cyan] {device}")
    console.print(f"[cyan]Text:[/cyan] {text[:80]}{'...' if len(text) > 80 else ''}\n")

    from .tracer import trace_model

    result = trace_model(
        text=text,
        model_id=model,
        quant=quant,
        device=device,
        backend_url=backend_url if not output else None,
        trust_remote_code=trust_remote_code,
    )

    if output:
        Path(output).write_text(result.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[green]Trace saved to {output}[/green]")
    elif result.trace_id is not None:
        console.print(f"[green]Trace sent to {backend_url} — ID {result.trace_id}[/green]")
    else:
        console.print(f"[yellow]Trace completed but backend upload failed[/yellow]")


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--model", "-m", default=None, help="HuggingFace model ID")
@click.option("--quant", "-q", default=None, help="Quantization")
@click.option("--device", "-d", default=None, help="Device")
@click.option("--backend", "-b", default=None, help="Backend URL")
@click.option(
    "--trust-remote-code",
    is_flag=True,
    default=None,
    help="Allow HuggingFace trust_remote_code (required for some MoE models)",
)
def batch(input_file, model, quant, device, backend, trust_remote_code):
    """Trace multiple texts from a file (one per line)."""
    _banner()
    config = get_config()

    texts = Path(input_file).read_text(encoding="utf-8").strip().split("\n")
    texts = [t.strip() for t in texts if t.strip()]

    model = model or config.default_model
    quant = quant or config.default_quantization
    device = device or config.default_device
    backend_url = backend or config.backend_url

    console.print(f"[cyan]Batch tracing {len(texts)} texts...[/cyan]\n")

    from .batch_tracer import batch_trace

    results = batch_trace(
        texts=texts,
        model_id=model,
        quant=quant,
        device=device,
        backend_url=backend_url,
        trust_remote_code=trust_remote_code,
    )

    table = Table(title="Batch Trace Results")
    table.add_column("#", style="cyan")
    table.add_column("Tokens", style="green")
    table.add_column("Trace ID", style="yellow")
    table.add_column("Status", style="bold")

    for i, r in enumerate(results):
        table.add_row(
            str(i + 1),
            str(r.get("num_tokens", "?")),
            str(r.get("trace_id", "?")),
            "[green]OK[/green]" if r.get("trace_id") else "[red]FAIL[/red]",
        )

    console.print(table)


@cli.command()
@click.option("--backend", "-b", default=None, help="Backend URL")
@click.option("--browser/--no-browser", default=True, help="Open browser automatically")
def viz(backend, browser):
    """Open the 3D visualizer in your browser."""
    config = get_config()
    url = backend or config.backend_url
    viz_url = f"{url}/visualizer"

    console.print(f"[cyan]Visualizer:[/cyan] {viz_url}")

    if browser:
        webbrowser.open(viz_url)
        console.print("[green]Opened in browser[/green]")
    else:
        console.print(f"[dim]Open manually: {viz_url}[/dim]")


@cli.command()
@click.option("--dir", "-d", default=None, help="Export directory")
@click.option("--format", "-f", default="json", type=click.Choice(["json", "jsonl", "parquet"]))
@click.option(
    "--append",
    "append_to",
    default=None,
    type=click.Path(),
    help="Append traces to this JSONL file (deduplicated) instead of writing a new file",
)
def export(dir, format, append_to):
    """Export all traces from the local database.

    With --append FILE your traces are added to a shared community JSONL file,
    skipping any whose routing data is already present.
    """
    config = get_config()

    if append_to:
        from .exporter import append_traces, collect_traces

        result = append_traces(collect_traces(config.db_path), append_to, dedup=True)
        console.print(
            f"[green]Appended {result.added} trace(s) to {result.path}[/green] "
            f"[dim]({result.duplicates} duplicate, {result.invalid} invalid skipped)[/dim]"
        )
        return

    export_dir = Path(dir) if dir else config.export_dir
    export_dir.mkdir(parents=True, exist_ok=True)

    from .exporter import export_traces

    output_path = export_traces(
        db_path=config.db_path,
        output_dir=export_dir,
        fmt=format,
    )

    console.print(f"[green]Exported to {output_path}[/green]")


@cli.command()
@click.argument("input_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--output", "-o", required=True, type=click.Path(), help="Combined output JSONL file")
@click.option(
    "--dedup/--no-dedup",
    default=True,
    help="Skip traces with identical routing data (default: on)",
)
def merge(input_files, output, dedup):
    """Merge trace files into one shared JSONL file.

    Example: moe-atlas merge traces1.jsonl traces2.jsonl --output combined.jsonl

    Records are validated and (by default) deduplicated. The output may also be
    one of the inputs, so a canonical community file can grow in place.
    """
    from .exporter import merge_trace_files

    result = merge_trace_files(list(input_files), output, dedup=dedup)
    console.print(
        f"[green]Merged {result.inputs} file(s) -> {result.output}[/green]\n"
        f"[dim]{result.written} written, {result.duplicates} duplicate, "
        f"{result.invalid} invalid skipped[/dim]"
    )


@cli.command()
@click.argument("input_files", nargs=-1, required=True, type=click.Path(exists=True))
def validate(input_files):
    """Validate trace file(s) before sharing or contributing.

    Reports valid, duplicate, and invalid records per file, and exits non-zero
    if any record is invalid -- handy in CI for community trace pull requests.
    """
    from .exporter import validate_trace_file

    total_invalid = 0
    for input_file in input_files:
        result = validate_trace_file(input_file)
        total_invalid += result.invalid
        color = "red" if result.invalid else "green"
        console.print(
            f"[{color}]{result.path}[/{color}]: {result.valid} valid, "
            f"{result.duplicates} duplicate, {result.invalid} invalid"
        )
        for err in result.errors:
            console.print(f"  [dim]{err}[/dim]")

    if total_invalid:
        raise SystemExit(1)


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--backend", "-b", default=None, help="Backend URL to import into")
def import_traces(input_file, backend):
    """Import traces from a file into the backend database."""
    config = get_config()
    backend_url = backend or config.backend_url

    from .exporter import import_traces

    count = import_traces(input_file, backend_url)
    console.print(f"[green]Imported {count} traces into {backend_url}[/green]")


@cli.command()
def init():
    """Initialize the MoE Atlas configuration directory."""
    config = get_config()
    config.db_path.parent.mkdir(parents=True, exist_ok=True)
    config.export_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[green]Initialized MoE Atlas config[/green]")
    console.print(f"[dim]Database:[/dim] {config.db_path}")
    console.print(f"[dim]Exports:[/dim] {config.export_dir}")
    console.print(f"[dim]Config env prefix:[/dim] MOE_ATLAS_")


@cli.command()
def config_show():
    """Show current configuration."""
    config = get_config()

    table = Table(title="MoE Atlas Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    for key, value in config.model_dump().items():
        table.add_row(key, str(value))

    console.print(table)


@cli.command()
@click.argument("input_file", required=False, type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Write tagged JSONL here (file mode)")
@click.option(
    "--db",
    "use_db",
    is_flag=True,
    help="Categorize traces in the local database (adds a 'domain' column)",
)
@click.option(
    "--endpoint",
    "-e",
    default=None,
    help="OpenAI-compatible chat endpoint for the classifier LLM",
)
@click.option("--model", "-m", default=None, help="Model name passed to the endpoint")
@click.option(
    "--min-trace-id",
    default=None,
    type=int,
    help="Only categorize traces with trace_id greater than this (db mode)",
)
@click.option("--batch-size", default=12, type=int, help="Texts per classification request")
def categorize(input_file, output, use_db, endpoint, model, min_trace_id, batch_size):
    """Auto-categorize traces into subject domains using an LLM endpoint.

    File mode:  moe-atlas categorize traces.jsonl -o tagged.jsonl -e http://localhost:8899
    DB mode:    moe-atlas categorize --db -e http://localhost:8899 [--min-trace-id N]

    Each trace gains a ``domain`` label; routing data is never modified. The
    endpoint may be any OpenAI-compatible chat server (set it with -e or the
    MOE_ATLAS_CLASSIFIER_ENDPOINT environment variable).
    """
    from .categorizer import categorize_db, categorize_file, make_llm_classifier

    config = get_config()
    endpoint = endpoint or config.classifier_endpoint
    if not endpoint:
        raise click.UsageError(
            "a classifier endpoint is required (use -e or set MOE_ATLAS_CLASSIFIER_ENDPOINT)"
        )
    classify_batch = make_llm_classifier(endpoint, model=model or config.classifier_model or None)

    if use_db:
        result = categorize_db(
            config.db_path, classify_batch, batch_size=batch_size, min_trace_id=min_trace_id
        )
        console.print(
            f"[green]Categorized {result.categorized}/{result.total} trace(s)[/green] "
            f"[dim]in {config.db_path}[/dim]"
        )
    else:
        if not input_file:
            raise click.UsageError("provide an INPUT_FILE (or use --db)")
        if not output:
            raise click.UsageError("--output/-o is required in file mode")
        result = categorize_file(input_file, output, classify_batch, batch_size=batch_size)
        console.print(
            f"[green]Categorized {result.categorized}/{result.total} trace(s) -> {output}[/green]"
        )

    table = Table(title="Domain distribution")
    table.add_column("Domain", style="cyan")
    table.add_column("Traces", justify="right", style="green")
    for domain, count in sorted(result.distribution.items(), key=lambda kv: (-kv[1], kv[0])):
        table.add_row(domain or "(none)", str(count))
    console.print(table)


# Entry point
def main():
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
