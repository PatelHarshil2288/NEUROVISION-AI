from __future__ import annotations

import errno
import os


def process_is_running(process_id: int) -> bool:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        synchronize = 0x00100000
        wait_object_0 = 0
        wait_timeout = 258
        invalid_parameter = 87
        access_denied = 5

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        open_process = kernel.OpenProcess
        open_process.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        open_process.restype = wintypes.HANDLE
        handle = open_process(synchronize, False, process_id)
        if not handle:
            error = ctypes.get_last_error()
            if error == access_denied:
                return True
            if error == invalid_parameter:
                return False
            raise OSError(error, ctypes.FormatError(error))
        try:
            wait_for_single_object = kernel.WaitForSingleObject
            wait_for_single_object.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            wait_for_single_object.restype = wintypes.DWORD
            result = wait_for_single_object(handle, 0)
            if result == wait_timeout:
                return True
            if result == wait_object_0:
                return False
            error = ctypes.get_last_error()
            raise OSError(error, ctypes.FormatError(error))
        finally:
            close_handle = kernel.CloseHandle
            close_handle.argtypes = (wintypes.HANDLE,)
            close_handle.restype = wintypes.BOOL
            close_handle(handle)

    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        if exc.errno == errno.EPERM:
            return True
        raise
    return True
