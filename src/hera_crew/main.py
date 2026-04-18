import sys
import os
import asyncio

src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_path not in sys.path:
    sys.path.append(src_path)

from hera_crew.utils.env_setup import setup_environment
from hera_crew.crew import HeraCrew

from rich.console import Console
from rich.panel import Panel

console = Console()


async def main():
    setup_environment()

    user_request = input("Please enter your development task (Japanese/English): ")
    if not user_request:
        user_request = "Create a simple Python script for calculating prime numbers."

    try:
        crew = HeraCrew()
        result = await crew.run(user_request)
        console.print(
            Panel(
                result,
                title="[bold green]✅ HERA Final Output[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
