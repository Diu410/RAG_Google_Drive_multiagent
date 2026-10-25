"""Shared file revisions for metadata scans and extracted-text caches.

On Windows st_ctime is creation time. FILE_BASIC_INFO.ChangeTime observes the
metadata update even when a synchronizer restores LastWriteTime after a write.
Unsupported Windows filesystems use a content digest instead of a weak cache key.
"""
import hashlib
import os
import sys

if sys.platform == 'win32':
    import ctypes
    import msvcrt
    from ctypes import wintypes

    class _FileBasicInfo(ctypes.Structure):
        _fields_ = [(name, ctypes.c_longlong) for name in
                    ('creation', 'access', 'write', 'change')] + [('attributes', wintypes.DWORD)]

    _get_info = ctypes.WinDLL('kernel32', use_last_error=True).GetFileInformationByHandleEx
    _get_info.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    _get_info.restype = wintypes.BOOL


def _change_time(stream, stat):
    if sys.platform != 'win32':
        return stat.st_ctime_ns
    info = _FileBasicInfo()
    if _get_info(msvcrt.get_osfhandle(stream.fileno()), 0, ctypes.byref(info), ctypes.sizeof(info)):
        return info.change or None
    return None


def file_revision(path):
    with path.open('rb') as stream:
        stat = os.fstat(stream.fileno())
        change = _change_time(stream, stat)
        if change is None:
            digest = hashlib.blake2b(digest_size=20)
            while block := stream.read(256 * 1024):
                digest.update(block)
            change = digest.hexdigest()
        return f'{stat.st_dev}:{stat.st_ino}:{stat.st_size}:{stat.st_mtime_ns}:{change}'
