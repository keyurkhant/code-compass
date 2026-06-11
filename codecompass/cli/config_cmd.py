"""codecompass config — read and write settings in ~/.config/codecompass/config.toml"""
import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group("config")
def config_group() -> None:
    """Manage code-compass configuration."""


@config_group.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    """Set a config value. KEY uses dot notation: llm.provider, llm.model, etc.

    \b
    Examples:
        codecompass config set llm.provider claude-code
        codecompass config set llm.provider ollama
        codecompass config set llm.model llama3.2
        codecompass config set llm.base_url http://localhost:11434
        codecompass config set embedding.model BAAI/bge-small-en-v1.5
    """
    from codecompass.config.manager import set_config_value

    set_config_value(key, value)
    console.print(f"[green]✓[/green] Set [cyan]{key}[/cyan] = [yellow]{value}[/yellow]")


@config_group.command("get")
@click.argument("key")
def config_get(key: str) -> None:
    """Get a config value by dot-notation key."""
    from codecompass.config.manager import get_config_value

    value = get_config_value(key)
    if value:
        console.print(f"[cyan]{key}[/cyan] = [yellow]{value}[/yellow]")
    else:
        console.print(f"[dim]{key} is not set (using default)[/dim]")


@config_group.command("list")
def config_list() -> None:
    """Show all current configuration values."""
    import dataclasses

    from codecompass.config.manager import CONFIG_PATH, load_config

    cfg = load_config()

    table = Table(title=f"code-compass config ({CONFIG_PATH})", show_header=True)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="yellow")

    def flatten(d: dict, prefix: str = "") -> list[tuple[str, str]]:
        items = []
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.extend(flatten(v, full_key))
            else:
                items.append((full_key, str(v) if v else "[dim](default)[/dim]"))
        return items

    for key, value in flatten(dataclasses.asdict(cfg)):
        table.add_row(key, value)

    console.print(table)


@config_group.command("unset")
@click.argument("key")
def config_unset(key: str) -> None:
    """Remove a config value (revert to default)."""
    from codecompass.config.manager import set_config_value

    set_config_value(key, "")
    console.print(f"[yellow]Unset[/yellow] [cyan]{key}[/cyan] (will use default)")
