"""MoE Atlas CLI — command-line interface for the routing atlas toolkit."""

import subprocess
import sys
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
@click.option("--host", default=None, help="Server host (default: 0.0.0.0)")
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
def trace(text, model, quant, device, backend, file, output):
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
    )

    if output:
        Path(output).write_text(result.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[green]Trace saved to {output}[/green]")
    else:
        console.print(f"[green]Trace sent to {backend_url} — ID {result.trace_id}[/green]")


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--model", "-m", default=None, help="HuggingFace model ID")
@click.option("--quant", "-q", default=None, help="Quantization")
@click.option("--device", "-d", default=None, help="Device")
@click.option("--backend", "-b", default=None, help="Backend URL")
def batch(input_file, model, quant, device, backend):
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
def export(dir, format):
    """Export all traces from the local database."""
    config = get_config()
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


# Entry point
def main():
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
