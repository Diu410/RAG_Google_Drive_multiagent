"""Check tracked source files without ever printing a matched credential."""
from pathlib import Path
import re
import subprocess
import sys

PATTERNS = (
    re.compile(rb'AIza[0-9A-Za-z_-]{35}'),
    re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----\s+[A-Za-z0-9+/]{30,}'),
)


def main():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(['git', 'ls-files', '-z'], cwd=root, capture_output=True, check=True)
    found = []
    for name in result.stdout.decode('utf-8').split('\0'):
        path = root / name
        if not name or not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
            continue
        content = path.read_bytes()
        if any(pattern.search(content) for pattern in PATTERNS):
            found.append(name)
    if found:
        print('Credential-like material found in tracked files; values are redacted:')
        print('\n'.join(found))
        return 1
    print('PASS: no credential material matched in tracked source files (Git history was not rewritten).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
