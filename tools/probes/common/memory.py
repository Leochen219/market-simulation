"""Dependency-free host memory metadata."""
from __future__ import annotations
import os
import sys
from typing import Any

def host_memory_bytes() -> dict[str, int | None]:
    if sys.platform == "win32":
        import ctypes
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return {"ram_total_bytes": int(status.ullTotalPhys), "ram_available_bytes": int(status.ullAvailPhys)}
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        physical_pages = os.sysconf("SC_PHYS_PAGES")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        return {"ram_total_bytes": int(page_size * physical_pages),
                "ram_available_bytes": int(page_size * available_pages)}
    except (AttributeError, OSError, ValueError):
        return {"ram_total_bytes": None, "ram_available_bytes": None}
