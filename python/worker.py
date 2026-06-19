from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import traceback
from typing import Any, Callable


PROTOCOL_VERSION = 1


class Stream:
    def read_exactly(self, size: int) -> bytes:
        raise NotImplementedError

    def write_all(self, data: bytes) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class SocketStream(Stream):
    def __init__(self, connection: socket.socket):
        self.connection = connection

    def read_exactly(self, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = self.connection.recv(remaining)
            if not chunk:
                raise EOFError
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def write_all(self, data: bytes) -> None:
        self.connection.sendall(data)

    def close(self) -> None:
        self.connection.close()


class WindowsPipeStream(Stream):
    def __init__(self, endpoint: str):
        import ctypes
        from ctypes import wintypes

        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32.CreateNamedPipeW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
        ]
        self._kernel32.CreateNamedPipeW.restype = wintypes.HANDLE
        self._kernel32.ConnectNamedPipe.argtypes = [
            wintypes.HANDLE,
            wintypes.LPVOID,
        ]
        self._kernel32.ConnectNamedPipe.restype = wintypes.BOOL
        self._kernel32.ReadFile.argtypes = [
            wintypes.HANDLE,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.LPVOID,
        ]
        self._kernel32.ReadFile.restype = wintypes.BOOL
        self._kernel32.WriteFile.argtypes = self._kernel32.ReadFile.argtypes
        self._kernel32.WriteFile.restype = wintypes.BOOL
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel32.DisconnectNamedPipe.argtypes = [wintypes.HANDLE]
        self._kernel32.FlushFileBuffers.argtypes = [wintypes.HANDLE]
        self._handle = self._kernel32.CreateNamedPipeW(
            endpoint,
            0x00000003,  # PIPE_ACCESS_DUPLEX
            0x00000000,  # PIPE_TYPE_BYTE | PIPE_READMODE_BYTE
            1,
            65536,
            65536,
            0,
            None,
        )
        if self._handle == wintypes.HANDLE(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())

        connected = self._kernel32.ConnectNamedPipe(self._handle, None)
        if not connected and ctypes.get_last_error() != 535:  # ERROR_PIPE_CONNECTED
            raise ctypes.WinError(ctypes.get_last_error())

    def read_exactly(self, size: int) -> bytes:
        import ctypes

        result = bytearray()
        while len(result) < size:
            buffer = ctypes.create_string_buffer(size - len(result))
            read = ctypes.c_ulong()
            ok = self._kernel32.ReadFile(
                self._handle, buffer, len(buffer), ctypes.byref(read), None
            )
            if not ok or read.value == 0:
                raise EOFError
            result.extend(buffer.raw[: read.value])
        return bytes(result)

    def write_all(self, data: bytes) -> None:
        import ctypes

        offset = 0
        while offset < len(data):
            written = ctypes.c_ulong()
            chunk = data[offset:]
            buffer = ctypes.create_string_buffer(chunk)
            ok = self._kernel32.WriteFile(
                self._handle, buffer, len(chunk), ctypes.byref(written), None
            )
            if not ok:
                raise ctypes.WinError(ctypes.get_last_error())
            offset += written.value

    def close(self) -> None:
        self._kernel32.FlushFileBuffers(self._handle)
        self._kernel32.DisconnectNamedPipe(self._handle)
        self._kernel32.CloseHandle(self._handle)


def open_stream(endpoint: str) -> Stream:
    if os.name == "nt":
        return WindowsPipeStream(endpoint)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(endpoint)
        server.listen(1)
        connection, _ = server.accept()
    finally:
        server.close()
    return SocketStream(connection)


def read_message(stream: Stream) -> dict[str, Any]:
    size = struct.unpack(">I", stream.read_exactly(4))[0]
    return json.loads(stream.read_exactly(size).decode("utf-8"))


def write_message(stream: Stream, value: dict[str, Any]) -> None:
    body = json.dumps(value, separators=(",", ":")).encode("utf-8")
    stream.write_all(struct.pack(">I", len(body)) + body)


class TorchRuntime:
    def __init__(self) -> None:
        import torch

        self.torch = torch
        self.tensors: dict[int, Any] = {}
        self.next_handle = 1
        self.operations: dict[str, Callable[..., Any]] = {
            "squeeze": self._squeeze,
            "sum": self._reducer("sum"),
            "mean": self._reducer("mean"),
            "prod": self._prod,
            "nansum": self._reducer("nansum", allow_none_keepdim=True),
            "nanprod": self._nanprod,
        }

    def execute(
        self, op: str, args: list[Any], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        args = self._decode(args)
        kwargs = self._decode(kwargs)

        if op == "__init__":
            self._initialize(kwargs["default_dtype"], kwargs["default_device"])
            return {"kind": "initialized"}
        if op == "__release__":
            self.tensors.pop(args[0], None)
            return {"kind": "released"}

        function = self.operations.get(op) or getattr(self.torch, op)
        result = function(*args, **self._dtype_kwargs(kwargs))
        if not isinstance(result, self.torch.Tensor):
            raise TypeError(f"torch.{op} returned {type(result).__name__}, not Tensor")
        return self._store(result)

    def _initialize(self, default_dtype: str, default_device: str) -> None:
        dtype = getattr(self.torch, default_dtype)
        self.torch.set_default_dtype(dtype)
        if hasattr(self.torch, "set_default_device"):
            self.torch.set_default_device(default_device)
        elif default_device != "cpu":
            raise RuntimeError(
                "This PyTorch version does not support torch.set_default_device"
            )

    def _decode(self, value: Any) -> Any:
        if isinstance(value, list):
            return [self._decode(item) for item in value]
        if isinstance(value, dict):
            if set(value) == {"$tensor"}:
                handle = value["$tensor"]
                try:
                    return self.tensors[handle]
                except KeyError:
                    raise KeyError(f"Unknown tensor handle {handle}") from None
            return {key: self._decode(item) for key, item in value.items()}
        return value

    def _dtype_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        if "dtype" in kwargs:
            kwargs["dtype"] = getattr(self.torch, kwargs["dtype"])
        return kwargs

    def _squeeze(self, input: Any, dim: Any = None) -> Any:
        if isinstance(dim, list):
            dim = tuple(dim)
        if dim is None:
            return self.torch.squeeze(input)
        return self.torch.squeeze(input, dim=dim)

    def _reducer(
        self, name: str, allow_none_keepdim: bool = False
    ) -> Callable[..., Any]:
        def reduce(input: Any, dim: Any = None, **kwargs: Any) -> Any:
            function = getattr(self.torch, name)
            if isinstance(dim, list):
                dim = tuple(dim)
            if dim is not None:
                kwargs["dim"] = dim
            elif allow_none_keepdim and "keepdim" in kwargs:
                kwargs["dim"] = None
            else:
                kwargs.pop("keepdim", None)
            return function(input, **kwargs)

        return reduce

    def _nanprod(self, input: Any, dim: Any = None, **kwargs: Any) -> Any:
        if hasattr(self.torch, "nanprod"):
            function = self.torch.nanprod
            if isinstance(dim, list):
                dim = tuple(dim)
            if dim is not None or "keepdim" in kwargs:
                kwargs["dim"] = dim
            return function(input, **kwargs)
        input = self.torch.nan_to_num(input, nan=1.0)
        return self._prod(input, dim=dim, **kwargs)

    def _prod(
        self,
        input: Any,
        dim: Any = None,
        keepdim: bool = False,
        dtype: Any = None,
    ) -> Any:
        dtype_kwargs = {} if dtype is None else {"dtype": dtype}
        if dim is None:
            return self.torch.prod(input, **dtype_kwargs)
        if isinstance(dim, int):
            return self.torch.prod(
                input, dim=dim, keepdim=keepdim, **dtype_kwargs
            )

        dims = [item % input.dim() for item in dim]
        if not keepdim:
            dims.sort(reverse=True)
        result = input
        for current_dim in dims:
            result = self.torch.prod(
                result,
                dim=current_dim,
                keepdim=keepdim,
                **dtype_kwargs,
            )
        return result

    def _store(self, tensor: Any) -> dict[str, Any]:
        handle = self.next_handle
        self.next_handle += 1
        self.tensors[handle] = tensor
        return {
            "kind": "tensor",
            "handle": handle,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "device": str(tensor.device),
            "repr": repr(tensor),
        }


def serve(endpoint: str) -> None:
    stream = open_stream(endpoint)
    try:
        runtime = TorchRuntime()
        while True:
            request = read_message(stream)
            request_id = request["id"]
            if request["op"] == "__close__":
                break
            try:
                result = runtime.execute(
                    request["op"],
                    request.get("args", []),
                    request.get("kwargs", {}),
                )
                write_message(
                    stream, {"id": request_id, "ok": True, "result": result}
                )
            except Exception as error:
                write_message(
                    stream,
                    {
                        "id": request_id,
                        "ok": False,
                        "error": {
                            "type": type(error).__name__,
                            "message": str(error),
                            "traceback": traceback.format_exc(),
                        },
                    },
                )
    except EOFError:
        pass
    finally:
        stream.close()
        if os.name != "nt":
            try:
                os.unlink(endpoint)
            except FileNotFoundError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--protocol-version", type=int, required=True)
    args = parser.parse_args()
    if args.protocol_version != PROTOCOL_VERSION:
        raise RuntimeError(
            f"Protocol mismatch: Node requested {args.protocol_version}, "
            f"worker supports {PROTOCOL_VERSION}"
        )
    serve(args.endpoint)


if __name__ == "__main__":
    main()
