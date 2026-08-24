"""Computer-language problem generators (M19).

Same M08/M18 contract as the math corpus: ``gen(rng) -> Problem`` with
answer AND step-by-step solution computed in code - deterministic under a
seeded Random (M01 rule).  Three domains:

* ``programming``        - Python trace-the-output, complexity, core syntax
* ``machine_language``   - binary/hex/octal conversion, bitwise ops, shifts
* ``networking``         - IP/CIDR math, ports, protocols, HTTP semantics

These give the LLM supervised exposure to the languages machines speak,
following the identical lesson-graded pipeline as mathematics.
"""

from __future__ import annotations

import ipaddress
import random
from collections.abc import Callable, Mapping

from foundation.math_corpus.problems import Problem

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

WELL_KNOWN_PORTS = {
    20: "FTP data", 21: "FTP control", 22: "SSH", 23: "Telnet",
    25: "SMTP", 53: "DNS", 67: "DHCP server", 80: "HTTP",
    110: "POP3", 143: "IMAP", 443: "HTTPS", 3389: "RDP",
}

HTTP_STATUS = {
    200: "OK - the request succeeded",
    301: "Moved Permanently - the resource lives at a new URL",
    403: "Forbidden - the server refuses to authorize the request",
    404: "Not Found - no resource at this URL",
    500: "Internal Server Error - the server failed handling the request",
    503: "Service Unavailable - temporarily overloaded or down",
}


def _bin(x: int, bits: int = 8) -> str:
    return format(x & ((1 << bits) - 1), f"0{bits}b")


# ---------------------------------------------------------------------------
# programming
# ---------------------------------------------------------------------------

def _gen_programming(rng: random.Random) -> Problem:
    kind = rng.randrange(4)

    if kind == 0:  # trace arithmetic + list ops
        a, b, c = (rng.randint(2, 12) for _ in range(3))
        lst = [rng.randint(1, 9) for _ in range(4)]
        snippet = (
            "x = {a} * {b}\n"
            "nums = {lst}\n"
            "nums.append(x)\n"
            "print(len(nums), nums[0] + x)"
        ).format(a=a, b=b, c=c, lst=lst)
        out_len = len(lst) + 1
        out_sum = lst[0] + a * b
        return Problem(
            f"What does this Python code print?\n{snippet}",
            f"{out_len} {out_sum}",
            f"x = {a}*{b} = {a * b}. append makes the list length "
            f"{len(lst)}+1 = {out_len}. nums[0]+x = {lst[0]}+{a * b} = {out_sum}. "
            f"print outputs '{out_len} {out_sum}'.",
        )

    if kind == 1:  # string slicing
        word = rng.choice(["silverwing", "platform", "network", "machine", "decoder"])
        s = word[1:-1]
        idx = rng.randrange(1, max(2, len(word) - 2))
        return Problem(
            f'In Python, what is {word!r}[{idx}:{idx + 3}]?',
            repr(word[idx:idx + 3]),
            f"Slicing is half-open: characters at indices {idx}, {idx+1}, {idx+2} -> "
            f"'{word[idx:idx + 3]}'.",
        )

    if kind == 2:  # loop result / sum
        n = rng.randint(3, 7)
        step = rng.choice([2, 3])
        total = sum(range(0, n * step, step))
        seq = ", ".join(str(i) for i in range(0, n * step, step))
        return Problem(
            f"What is the value of total after:\ntotal = 0\nfor i in range(0, {n * step}, {step}):\n    total += i",
            str(total),
            f"range yields {seq}; their sum is {total}.",
        )

    # kind == 3: big-O recognition
    which = rng.randrange(3)
    if which == 0:
        q = ("What is the time complexity of nested loops where the outer runs n times "
             "and the inner also runs n times?")
        ans = "O(n^2)"
        sol = "n iterations x n iterations each = n*n = n^2 operations."
    elif which == 1:
        q = ("What is the time complexity of binary search on a sorted array of n elements?")
        ans = "O(log n)"
        sol = "Each comparison halves the search space: n -> n/2 -> ... -> 1 takes log2(n) steps."
    else:
        q = "What is the time complexity of accessing a single element of a list by index?"
        ans = "O(1)"
        sol = "Lists are backed by arrays; index access computes an address directly - constant time."
    return Problem(q, ans, sol)


# ---------------------------------------------------------------------------
# machine language
# ---------------------------------------------------------------------------

def _gen_machine_language(rng: random.Random) -> Problem:
    kind = rng.randrange(4)

    if kind == 0:  # decimal -> binary
        x = rng.randint(5, 255)
        return Problem(
            f"Convert the decimal number {x} to 8-bit binary.",
            _bin(x),
            f"Powers of two summing to {x}: "
            + " + ".join(f"2^{b}" for b in range(7, -1, -1) if x >> b & 1)
            + f", giving {_bin(x)}.",
        )

    if kind == 1:  # binary -> decimal
        x = rng.randint(5, 255)
        b = _bin(x)
        terms = " + ".join(f"{int(bit)}*2^{7 - i}" for i, bit in enumerate(b) if bit == "1")
        return Problem(
            f"Convert the binary number {b} to decimal.",
            str(x),
            f"{terms} = {x}.",
        )

    if kind == 2:  # hex <-> decimal
        x = rng.randint(16, 255)
        h = format(x, "X")
        return Problem(
            f"Convert hexadecimal 0x{h} to decimal.",
            str(x),
            f"0x{h} = {x} in decimal (hex digit place values are powers of 16).",
        )

    # kind == 3: bitwise op on two bytes
    op_name, fn = rng.choice([
        ("AND", lambda p, q: p & q),
        ("OR", lambda p, q: p | q),
        ("XOR", lambda p, q: p ^ q),
    ])
    a, b = rng.randint(8, 200), rng.randint(8, 200)
    r = fn(a, b)
    return Problem(
        f"Compute the bitwise {op_name} of {a} and {b} ({a} {op_name} {b}).",
        str(r),
        f"{a} = {_bin(a)}, {b} = {_bin(b)}. Applying {op_name} bit-by-bit gives "
        f"{_bin(r)} = {r}.",
    )


# ---------------------------------------------------------------------------
# networking
# ---------------------------------------------------------------------------

def _gen_networking(rng: random.Random) -> Problem:
    kind = rng.randrange(4)

    if kind == 0:  # /24 host count
        prefix = rng.choice([24, 25, 26, 28])
        hosts = 2 ** (32 - prefix) - 2
        return Problem(
            f"How many usable host addresses does the IPv4 subnet /{prefix} provide?",
            str(hosts),
            f"Host bits = 32-{prefix} = {32 - prefix}, so 2^{32 - prefix} = {2 ** (32 - prefix)} "
            f"total addresses; minus network and broadcast leaves {hosts} usable.",
        )

    if kind == 1:  # network address of an IP/prefix
        prefix = rng.choice([24, 25, 26])
        ip_int = rng.randint(0xC0A80000, 0xC0A800FF)  # 192.168.x.y space
        ip = str(ipaddress.IPv4Address(ip_int))
        net_obj = ipaddress.ip_network(f"{ip}/{prefix}", strict=False)
        net = str(net_obj.network_address)
        keep = prefix // 8
        mask = str(net_obj.netmask)
        return Problem(
            f"What is the network address of {ip}/{prefix}?",
            net,
            f"Mask /{prefix} = {mask} keeps the first {keep} octet(s) fixed and zeroes "
            f"the host bits, so the network address is {net}.",
        )

    if kind == 2:  # port lookup
        port, svc = rng.choice(list(WELL_KNOWN_PORTS.items()))
        return Problem(
            f"Which well-known service uses TCP port {port}?",
            svc,
            f"Port {port} is the standard port for {svc}.",
        )

    # HTTP status semantics
    code, meaning = rng.choice(list(HTTP_STATUS.items()))
    cls = "success" if 200 <= code < 300 else "redirection" if 300 <= code < 400 else \
          "client error" if 400 <= code < 500 else "server error"
    return Problem(
        f"In HTTP, what does status code {code} mean, and which class does it belong to?",
        meaning,
        f"Codes are classed by their first digit ({code} -> {cls}). {code} means: {meaning}.",
    )


COMPUTER_GENERATORS: Mapping[str, Callable[[random.Random], Problem]] = {
    "programming": _gen_programming,
    "machine_language": _gen_machine_language,
    "networking": _gen_networking,
}
