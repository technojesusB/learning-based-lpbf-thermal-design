import subprocess
import sys


def run_command(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)


def lint() -> None:
    """Run ruff check."""
    print("Running lint (ruff check)...")
    run_command(["ruff", "check", "."])


def format() -> None:
    """Run ruff format --check."""
    print("Running format check (ruff format --check)...")
    run_command(["ruff", "format", "--check", "."])


def typecheck() -> None:
    """Run pyright."""
    print("Running typecheck (pyright)...")
    run_command(["pyright"])


def test() -> None:
    """Run pytest with arguments."""
    args = sys.argv[1:]
    print(f"Running tests (pytest {' '.join(args)})...")
    run_command(["pytest"] + args)
