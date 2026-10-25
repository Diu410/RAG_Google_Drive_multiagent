"""Синтетическая политика таймаутов. Файл предназначен для чтения, не запуска."""
DEFAULT_TIMEOUT_SECONDS = 45
PER_SERVICE_TIMEOUT_SECONDS = {'archive': 12, 'search': 18}


def timeout_seconds(service, override=None):
    if override is not None:
        return override
    return PER_SERVICE_TIMEOUT_SECONDS.get(service, DEFAULT_TIMEOUT_SECONDS)
