"""Shared exclusions for browsing, retrieval, indexing and system opening."""
from pathlib import Path
import fnmatch
import re


class FilePolicyError(PermissionError):
    """A safe policy explanation that can be shown to the user."""

SECRET_PATTERNS = (
    '.env', '.env.*', 'token.json', 'token.*.json', 'credentials.json', 'credentials.*.json',
    'client_secret*.json', 'service_account*.json', 'service-account*.json',
    '*.pem', '*.key', '*.p12', '*.pfx', 'id_rsa*', 'id_ed25519*',
    'application_default_credentials.json', 'authorized_user.json', 'secrets.json', 'keys.json',
)
SAFE_OPEN_EXTENSIONS = {
    '.pdf', '.docx', '.txt', '.md', '.markdown', '.csv', '.tsv', '.rtf',
    '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tif', '.tiff',
    '.xlsx', '.xls', '.pptx', '.ppt', '.odt', '.ods', '.odp',
}


def is_secret_file(path):
    name = Path(path).name.casefold()
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in SECRET_PATTERNS)


def has_credential_content(path):
    path = Path(path)
    if path.suffix.casefold() not in {'.json', '.ini', '.conf', '.cfg', '.yaml', '.yml', '.toml'}:
        return False
    try:
        with path.open('rb') as stream:
            prefix = stream.read(65536).decode('utf-8', errors='ignore')
    except OSError:
        return False
    return bool(re.search(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|'
        r'["\'](?:private_key|refresh_token|access_token|client_secret|aws_secret_access_key)["\']\s*:\s*["\'][^"\']{12,}',
        prefix, re.I))


def can_open_file(path):
    return not is_secret_file(path) and Path(path).suffix.casefold() in SAFE_OPEN_EXTENSIONS
