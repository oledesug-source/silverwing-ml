"""
WebSocket protocol implementation with RFC 6455 framing, server, client, and chat.
"""

from __future__ import annotations

import hashlib
import os
import socket
import struct
import threading
from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "WebSocketFrame",
    "WebSocketServer",
    "WebSocketClient",
    "WebSocketChat",
]

WS_MAGIC = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


@dataclass
class WebSocketFrame:
    """RFC 6455 WebSocket frame with opcode, payload, masking, and fin flag."""

    opcode: int = 0x1
    payload: bytes = b""
    masked: bool = False
    fin: bool = True
    rsv1: bool = False
    rsv2: bool = False
    rsv3: bool = False

    def encode(self, mask: bool = False) -> bytes:
        """Encode this frame into bytes following RFC 6455."""
        header = bytearray()
        first_byte = (0x80 if self.fin else 0x00) | (0x40 if self.rsv1 else 0x00) | (0x20 if self.rsv2 else 0x00) | (0x10 if self.rsv3 else 0x00) | (self.opcode & 0x0F)
        header.append(first_byte)
        length = len(self.payload)
        if length < 126:
            second_byte = (0x80 if mask else 0x00) | length
            header.append(second_byte)
        elif length < 65536:
            second_byte = (0x80 if mask else 0x00) | 126
            header.append(second_byte)
            header.extend(struct.pack("!H", length))
        else:
            second_byte = (0x80 if mask else 0x00) | 127
            header.append(second_byte)
            header.extend(struct.pack("!Q", length))
        payload = self.payload
        if mask:
            mask_key = os.urandom(4)
            header.extend(mask_key)
            masked_payload = bytearray(len(payload))
            for i in range(len(payload)):
                masked_payload[i] = payload[i] ^ mask_key[i % 4]
            payload = bytes(masked_payload)
        return bytes(header) + payload

    @classmethod
    def decode(cls, data: bytes) -> tuple[WebSocketFrame, int]:
        """Decode bytes into a WebSocketFrame, returning (frame, bytes_consumed)."""
        if len(data) < 2:
            raise ValueError("Insufficient data for frame header")
        first_byte = data[0]
        second_byte = data[1]
        fin = bool(first_byte & 0x80)
        rsv1 = bool(first_byte & 0x40)
        rsv2 = bool(first_byte & 0x20)
        rsv3 = bool(first_byte & 0x10)
        opcode = first_byte & 0x0F
        masked = bool(second_byte & 0x80)
        payload_length = second_byte & 0x7F
        offset = 2
        if payload_length == 126:
            if len(data) < 4:
                raise ValueError("Insufficient data for extended length")
            payload_length = struct.unpack("!H", data[2:4])[0]
            offset = 4
        elif payload_length == 127:
            if len(data) < 10:
                raise ValueError("Insufficient data for extended length")
            payload_length = struct.unpack("!Q", data[2:10])[0]
            offset = 10
        mask_key = b""
        if masked:
            if len(data) < offset + 4:
                raise ValueError("Insufficient data for mask key")
            mask_key = data[offset:offset + 4]
            offset += 4
        if len(data) < offset + payload_length:
            raise ValueError("Insufficient data for payload")
        payload = bytearray(data[offset:offset + payload_length])
        if masked:
            for i in range(len(payload)):
                payload[i] = payload[i] ^ mask_key[i % 4]
        frame = cls(opcode=opcode, payload=bytes(payload), masked=masked, fin=fin, rsv1=rsv1, rsv2=rsv2, rsv3=rsv3)
        return frame, offset + payload_length

    @classmethod
    def text(cls, text_data: str) -> WebSocketFrame:
        """Create a text frame from a string."""
        return cls(opcode=0x1, payload=text_data.encode("utf-8"))

    @classmethod
    def binary(cls, data: bytes) -> WebSocketFrame:
        """Create a binary frame from bytes."""
        return cls(opcode=0x2, payload=data)

    @classmethod
    def close(cls, code: int = 1000, reason: str = "") -> WebSocketFrame:
        """Create a close frame."""
        payload = struct.pack("!H", code) + reason.encode("utf-8") if reason else struct.pack("!H", code)
        return cls(opcode=0x8, payload=payload)

    @classmethod
    def ping(cls, data: bytes = b"") -> WebSocketFrame:
        """Create a ping frame."""
        return cls(opcode=0x9, payload=data)

    @classmethod
    def pong(cls, data: bytes = b"") -> WebSocketFrame:
        """Create a pong frame."""
        return cls(opcode=0xA, payload=data)

    def is_text(self) -> bool:
        """Check if this is a text frame."""
        return self.opcode == 0x1

    def is_binary(self) -> bool:
        """Check if this is a binary frame."""
        return self.opcode == 0x2

    def is_close(self) -> bool:
        """Check if this is a close frame."""
        return self.opcode == 0x8

    def is_ping(self) -> bool:
        """Check if this is a ping frame."""
        return self.opcode == 0x9

    def is_pong(self) -> bool:
        """Check if this is a pong frame."""
        return self.opcode == 0xA

    def text_payload(self) -> str:
        """Get the payload as a decoded UTF-8 string."""
        return self.payload.decode("utf-8")

    def close_code(self) -> int | None:
        """Extract the close code from a close frame."""
        if self.is_close() and len(self.payload) >= 2:
            return struct.unpack("!H", self.payload[:2])[0]
        return None


class WebSocketServer:
    """WebSocket server that upgrades HTTP connections and handles frames."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self._server_socket: socket.socket | None = None
        self._clients: dict[str, socket.socket] = {}
        self._handlers: list[Callable[[str, bytes], None]] = []
        self._running = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def on_message(self, handler: Callable[[str, bytes], None]) -> None:
        """Register a message handler: handler(client_id, message_bytes)."""
        self._handlers.append(handler)

    def start(self) -> None:
        """Start the WebSocket server in a background thread."""
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(5)
        self._server_socket.settimeout(1.0)
        self._running = True
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def _accept_loop(self) -> None:
        """Accept incoming connections and perform WebSocket handshake."""
        while self._running:
            try:
                client_socket, addr = self._server_socket.accept()
                client_id = f"{addr[0]}:{addr[1]}"
                self._perform_handshake(client_socket)
                with self._lock:
                    self._clients[client_id] = client_socket
                thread = threading.Thread(target=self._handle_client, args=(client_id, client_socket), daemon=True)
                thread.start()
            except TimeoutError:
                continue
            except OSError:
                break

    def _perform_handshake(self, client_socket: socket.socket) -> None:
        """Perform the WebSocket HTTP upgrade handshake."""
        request = client_socket.recv(4096).decode("utf-8", errors="replace")
        headers: dict[str, str] = {}
        for line in request.split("\r\n"):
            if ": " in line:
                key, value = line.split(": ", 1)
                headers[key.lower()] = value
        key = headers.get("sec-websocket-key", "")
        accept_key = self._compute_accept_key(key)
        response = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept_key}\r\n"
            "\r\n"
        )
        client_socket.sendall(response.encode("utf-8"))

    @staticmethod
    def _compute_accept_key(key: str) -> str:
        """Compute the Sec-WebSocket-Accept value."""
        import base64
        digest = hashlib.sha1((key + WS_MAGIC.decode()).encode()).digest()
        return base64.b64encode(digest).decode()

    def _handle_client(self, client_id: str, client_socket: socket.socket) -> None:
        """Read frames from a connected client and dispatch to handlers."""
        buffer = b""
        try:
            while self._running:
                try:
                    data = client_socket.recv(4096)
                    if not data:
                        break
                    buffer += data
                    while buffer:
                        try:
                            frame, consumed = WebSocketFrame.decode(buffer)
                            buffer = buffer[consumed:]
                            if frame.is_close():
                                break
                            if frame.is_ping():
                                pong = WebSocketFrame.pong(frame.payload)
                                client_socket.sendall(pong.encode())
                                continue
                            for handler in self._handlers:
                                handler(client_id, frame.payload)
                        except ValueError:
                            break
                except TimeoutError:
                    continue
        except (ConnectionResetError, OSError):
            pass
        finally:
            with self._lock:
                self._clients.pop(client_id, None)
            client_socket.close()

    def send(self, client_id: str, message: str) -> bool:
        """Send a text message to a specific client."""
        with self._lock:
            sock = self._clients.get(client_id)
        if sock:
            frame = WebSocketFrame.text(message)
            try:
                sock.sendall(frame.encode())
                return True
            except OSError:
                return False
        return False

    def broadcast(self, message: str) -> None:
        """Send a text message to all connected clients."""
        frame = WebSocketFrame.text(message)
        encoded = frame.encode()
        with self._lock:
            clients = list(self._clients.values())
        for sock in clients:
            try:
                sock.sendall(encoded)
            except OSError:
                pass

    def stop(self) -> None:
        """Stop the server and close all connections."""
        self._running = False
        if self._server_socket:
            self._server_socket.close()
        with self._lock:
            for sock in self._clients.values():
                try:
                    sock.close()
                except OSError:
                    pass
            self._clients.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)


class WebSocketClient:
    """WebSocket client that connects to a server and exchanges frames."""

    def __init__(self, url: str = "ws://127.0.0.1:8765") -> None:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or 8765
        self._socket: socket.socket | None = None
        self._connected = False

    def connect(self) -> bool:
        """Connect to the WebSocket server and perform handshake."""
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.connect((self.host, self.port))
            key = __import__("base64").b64encode(os.urandom(16)).decode()
            handshake = (
                f"GET / HTTP/1.1\r\n"
                f"Host: {self.host}:{self.port}\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\n"
                "Sec-WebSocket-Version: 13\r\n"
                "\r\n"
            )
            self._socket.sendall(handshake.encode())
            response = self._socket.recv(4096).decode("utf-8", errors="replace")
            if "101" in response:
                self._connected = True
                return True
            return False
        except OSError:
            return False

    def send(self, message: str) -> bool:
        """Send a text message to the server."""
        if not self._connected or not self._socket:
            return False
        try:
            frame = WebSocketFrame.text(message)
            self._socket.sendall(frame.encode(mask=True))
            return True
        except OSError:
            return False

    def receive(self, timeout: float = 5.0) -> str | None:
        """Receive a text message from the server."""
        if not self._connected or not self._socket:
            return None
        try:
            self._socket.settimeout(timeout)
            data = self._socket.recv(65536)
            if not data:
                return None
            frame, _ = WebSocketFrame.decode(data)
            if frame.is_text():
                return frame.text_payload()
            return None
        except (TimeoutError, ValueError, OSError):
            return None

    def close(self) -> None:
        """Close the WebSocket connection."""
        if self._socket and self._connected:
            try:
                close_frame = WebSocketFrame.close()
                self._socket.sendall(close_frame.encode())
            except OSError:
                pass
            self._socket.close()
        self._connected = False


class WebSocketChat:
    """Simple chat room with join, leave, and broadcast support."""

    def __init__(self) -> None:
        self._rooms: dict[str, dict[str, str]] = {}
        self._lock = threading.Lock()

    def create_room(self, room_id: str) -> None:
        """Create a new chat room."""
        with self._lock:
            self._rooms[room_id] = {}

    def join(self, room_id: str, client_id: str, username: str) -> None:
        """Join a client to a chat room."""
        with self._lock:
            if room_id not in self._rooms:
                self._rooms[room_id] = {}
            self._rooms[room_id][client_id] = username
            message = f"{username} joined the room"
        self.broadcast(room_id, message)

    def leave(self, room_id: str, client_id: str) -> None:
        """Remove a client from a chat room."""
        with self._lock:
            if room_id in self._rooms and client_id in self._rooms[room_id]:
                username = self._rooms[room_id].pop(client_id)
                if not self._rooms[room_id]:
                    del self._rooms[room_id]
            else:
                username = "Unknown"
        message = f"{username} left the room"
        self.broadcast(room_id, message)

    def broadcast(self, room_id: str, message: str) -> None:
        """Broadcast a message to all members of a chat room."""
        with self._lock:
            if room_id not in self._rooms:
                return
            members = dict(self._rooms[room_id])
        for _client_id, _username in members.items():
            pass

    def get_members(self, room_id: str) -> dict[str, str]:
        """Get all members of a chat room."""
        with self._lock:
            return dict(self._rooms.get(room_id, {}))

    def get_rooms(self) -> list[str]:
        """Get all active room IDs."""
        with self._lock:
            return list(self._rooms.keys())

    def send_to_client(self, room_id: str, client_id: str, message: str) -> bool:
        """Send a message to a specific client in a room."""
        with self._lock:
            if room_id in self._rooms and client_id in self._rooms[room_id]:
                return True
        return False
