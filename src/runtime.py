"""One persistent event loop for the whole Streamlit process.


This module exists because of a specific, expensive-to-debug failure:


    MCP stdio servers are child processes whose transports are bound to the
    event loop that created them. Streamlit re-runs the script top to bottom
    on every interaction. If each turn calls `asyncio.run()`, that creates a
    fresh loop, tears it down at the end, and kills the subprocess transports
    with it -- so the second message in a conversation fails with a closed
    pipe, or hangs.


The fix is to create exactly one loop, on a dedicated background thread, and
submit coroutines to it for the life of the process.
"""


from __future__ import annotations


import asyncio
import atexit
import sys
import threading
from concurrent.futures import Future
from typing import Any, Coroutine, TypeVar


T = TypeVar("T")


_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_lock = threading.Lock()




def _new_event_loop() -> asyncio.AbstractEventLoop:
    """Create a loop that can actually spawn subprocesses.


    Windows-specific trap. `asyncio.new_event_loop()` honours whatever event
    loop policy is installed, and Streamlit runs on Tornado, which installs
    `WindowsSelectorEventLoopPolicy`. A SelectorEventLoop on Windows raises


        NotImplementedError


    the moment anything tries to spawn a subprocess -- and MCP stdio servers
    are subprocesses. The failure surfaces from deep inside anyio's task group
    as "Unhandled errors in TaskGroup", which names neither the cause nor the
    platform.


    ProactorEventLoop is the Windows loop that supports subprocesses, so ask
    for it explicitly rather than accepting the policy's default. On every
    other platform the default loop already handles subprocesses.
    """
    if sys.platform == "win32":
        proactor = getattr(asyncio, "ProactorEventLoop", None)
        if proactor is not None:
            return proactor()
    return asyncio.new_event_loop()




def _start_loop() -> asyncio.AbstractEventLoop:
    loop = _new_event_loop()


    def runner() -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()


    thread = threading.Thread(target=runner, name="mcp-event-loop", daemon=True)
    thread.start()


    global _thread
    _thread = thread
    return loop




def get_loop() -> asyncio.AbstractEventLoop:
    """The process-wide event loop, started on first use."""
    global _loop
    with _lock:
        if _loop is None or _loop.is_closed():
            _loop = _start_loop()
        return _loop




def run_async(coro: Coroutine[Any, Any, T], timeout: float | None = 120.0) -> T:
    """Run a coroutine on the persistent loop and block until it finishes.


    Use this everywhere in the Streamlit app instead of `asyncio.run()`.
    """
    loop = get_loop()
    future: Future[T] = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout)




def shutdown() -> None:
    """Stop the loop. Registered atexit; rarely needed explicitly."""
    global _loop
    with _lock:
        if _loop is not None and not _loop.is_closed():
            _loop.call_soon_threadsafe(_loop.stop)
        _loop = None




atexit.register(shutdown)