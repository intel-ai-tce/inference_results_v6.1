# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0

"""HTTP socket binding for multi-plane endpoint fabrics."""

from __future__ import annotations

import os
import socket
from fnmatch import fnmatchcase
from collections.abc import Callable, Iterable
from http.client import HTTPResponse
from typing import Any


HTTP_INTERFACE_ENV = "MLPINF_HTTP_INTERFACE"
HTTP_INTERFACE_RULES_ENV = "MLPINF_HTTP_INTERFACE_RULES"


def configured_http_interface() -> str | None:
    interface = os.getenv(HTTP_INTERFACE_ENV, "").strip()
    return interface or None


def configured_http_interface_rules() -> tuple[tuple[str, str], ...]:
    rules = []
    for item in os.getenv(HTTP_INTERFACE_RULES_ENV, "").split(","):
        item = item.strip()
        if not item:
            continue
        pattern, separator, interface = item.partition("=")
        if not separator or not pattern.strip() or not interface.strip():
            raise ValueError(
                f"Invalid {HTTP_INTERFACE_RULES_ENV} entry: {item!r}"
            )
        rules.append((pattern.strip(), interface.strip()))
    return tuple(rules)


def http_interface_for_host(host: str, default: str | None = None) -> str | None:
    for pattern, interface in configured_http_interface_rules():
        if fnmatchcase(host, pattern):
            return interface
    return default or configured_http_interface()


def bind_to_device_socket_options(interface: str | None = None) -> list[tuple[int, int, bytes]]:
    interface = interface or configured_http_interface()
    if interface is None:
        return []
    return [(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, interface.encode() + b"\0")]


def aiohttp_socket_factory(interface: str | None = None) -> Callable[[Iterable[Any]], socket.socket] | None:
    if interface is None and configured_http_interface() is None and not configured_http_interface_rules():
        return None

    def create_socket(address_info: Iterable[Any]) -> socket.socket:
        family, socket_type, protocol, _, socket_address = address_info
        transport_socket = socket.socket(family, socket_type, protocol)
        destination_interface = http_interface_for_host(socket_address[0], interface)
        for level, option, value in bind_to_device_socket_options(destination_interface):
            transport_socket.setsockopt(level, option, value)
        return transport_socket

    return create_socket


def bound_http_health_check(
    endpoint: str,
    *,
    interface: str,
    timeout: float,
) -> None:
    host, separator, port_text = endpoint.rpartition(":")
    if not separator or not host:
        raise ValueError(f"Invalid HTTP endpoint: {endpoint!r}")
    port = int(port_text)
    interface = http_interface_for_host(host, interface)
    address_info = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)[0]
    family, socket_type, protocol, _, socket_address = address_info

    with socket.socket(family, socket_type, protocol) as transport_socket:
        for level, option, value in bind_to_device_socket_options(interface):
            transport_socket.setsockopt(level, option, value)
        transport_socket.settimeout(timeout)
        transport_socket.connect(socket_address)
        transport_socket.sendall(
            f"GET /health HTTP/1.1\r\nHost: {endpoint}\r\nConnection: close\r\n\r\n".encode()
        )
        response = HTTPResponse(transport_socket)
        response.begin()
        try:
            if not 200 <= response.status < 300:
                raise ConnectionError(
                    f"Health check returned HTTP {response.status} for {endpoint}"
                )
        finally:
            response.close()
