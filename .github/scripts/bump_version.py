import re
import sys


def bump_type(commit_message: str) -> str:
    first_line = commit_message.splitlines()[0].lower()

    if "breaking change" in commit_message.lower() or "!:" in first_line or "!)" in first_line:
        return "major"
    if first_line.startswith("feat"):
        return "minor"
    return "patch"


def main() -> None:
    commit_message = sys.argv[1]
    print(bump_type(commit_message))


if __name__ == "__main__":
    main()
