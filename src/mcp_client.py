"""Connect to the two MCP servers and expose their tools to LangChain.


Both servers are spawned as stdio subprocesses using the *same* interpreter
that runs the app, so a virtualenv is inherited without any PATH surprises.


The client is created once per process and kept alive; see src/runtime.py for
why the event loop must outlive individual turns.
"""


from __future__ import annotations


import os
import sys


from src.config import MCP_SERVERS_DIR, PROJECT_ROOT


SERVER_SPECS: dict[str, dict] = {
    "weather": {
        "command": sys.executable,
        "args": [str(MCP_SERVERS_DIR / "weather_server.py")],
        "transport": "stdio",
        "cwd": str(PROJECT_ROOT),
    },
    "currency": {
        "command": sys.executable,
        "args": [str(MCP_SERVERS_DIR / "currency_server.py")],
        "transport": "stdio",
        "cwd": str(PROJECT_ROOT),
    },
}


_client = None




def flatten_exception(exc: BaseException) -> list[BaseException]:
    """Unwrap nested ExceptionGroups down to the real leaf exceptions.

    The MCP stdio client runs inside an anyio task group, so a failure to
    launch a server arrives as `ExceptionGroup: unhandled errors in a
    TaskGroup (1 sub-exception)`. That string names neither the cause nor the
    file involved -- the actual error is one level down.
    """
    group = getattr(exc, "exceptions", None)
    if group and isinstance(exc, BaseException) and hasattr(exc, "exceptions"):
        leaves: list[BaseException] = []
        for inner in group:
            leaves.extend(flatten_exception(inner))
        return leaves or [exc]
    return [exc]




def describe_failure(exc: BaseException) -> str:
    leaves = flatten_exception(exc)
    detail = "; ".join(f"{type(e).__name__}: {e}" for e in leaves) or str(exc)


    hints: list[str] = []
    text = detail.lower()


    if any(isinstance(e, NotImplementedError) for e in leaves):
        hints.append(
            "NotImplementedError when spawning a subprocess on Windows means "
            "the event loop is a SelectorEventLoop, which cannot start child "
            "processes. src/runtime.py forces a ProactorEventLoop for exactly "
            "this reason -- check that run_async() is being used rather than "
            "asyncio.run()."
        )
    if "no such file" in text or "cannot find" in text or "winerror 2" in text:
        hints.append(
            f"The interpreter or server script could not be found. Expected "
            f"scripts under {MCP_SERVERS_DIR}."
        )
    if "zoneinfo" in text or "no time zone found" in text:
        hints.append(
            "ZoneInfo could not find the timezone database. On Windows that "
            "needs the `tzdata` package: pip install tzdata."
        )
    if any(isinstance(e, (ImportError, ModuleNotFoundError)) for e in leaves):
        hints.append(
            "A server crashed on import. Run it standalone to see the real "
            "traceback: python mcp_servers/weather_server.py"
        )


    if not hints:
        hints.append(
            "Run each server standalone to see its traceback -- they should "
            "start silently and wait on stdin:\n"
            "    python mcp_servers/weather_server.py\n"
            "    python mcp_servers/currency_server.py"
        )


    return f"{detail}\n\n" + "\n".join(f"- {h}" for h in hints)




def get_client():
    """The MultiServerMCPClient, created once per process."""
    global _client
    if _client is None:
        from langchain_mcp_adapters.client import MultiServerMCPClient


        _client = MultiServerMCPClient(SERVER_SPECS)
    return _client




async def load_mcp_tools() -> list:
    """Load tools from every configured MCP server.

    A server that fails to start is reported rather than raising: the app is
    still useful with RAG alone plus whichever server did come up, and
    pretending otherwise.
    """
    client = get_client()
    try:
        return await client.get_tools()
    except Exception as exc:  # noqa: BLE001
        raise MCPConnectionError(
            f"Could not load MCP tools.\n\n{describe_failure(exc)}"
        ) from exc




async def probe_servers() -> dict[str, dict]:
    """Per-server health, for the sidebar's connection indicators."""
    from langchain_mcp_adapters.client import MultiServerMCPClient


    status: dict[str, dict] = {}
    for name, spec in SERVER_SPECS.items():
        try:
            tools = await MultiServerMCPClient({name: spec}).get_tools()
            status[name] = {
                "ok": True,
                "tools": [t.name for t in tools],
            }
        except Exception as exc:  # noqa: BLE001
            status[name] = {"ok": False, "error": describe_failure(exc)}
    return status




class MCPConnectionError(RuntimeError):
    """Raised when the MCP servers cannot be reached at all."""




def _diagnose() -> int:
    """Standalone check: `python -m src.mcp_client`.

    Runs outside Streamlit, so it isolates whether a failure is in the MCP
    servers themselves or in how the app drives them.
    """
    import asyncio
    import platform
    import traceback


    from src.runtime import get_loop, run_async


    print("=" * 62)
    print("MCP server diagnostic")
    print("=" * 62)
    print(f"Python      : {sys.version.split()[0]} ({platform.system()})")
    print(f"Interpreter : {sys.executable}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Servers dir : {MCP_SERVERS_DIR}")


    loop = get_loop()
    print(f"Event loop  : {type(loop).__name__}")
    if sys.platform == "win32" and "Proactor" not in type(loop).__name__:
        print("  !! On Windows this loop cannot spawn subprocesses.")
    print(f"Policy      : {type(asyncio.get_event_loop_policy()).__name__}")
    print()


    for name, spec in SERVER_SPECS.items():
        script = spec["args"][0]
        exists = "found" if os.path.exists(script) else "MISSING"
        print(f"[{name}] {script}  ({exists})")


    # Launch each server as a plain subprocess first. A server that crashes or
    # exits during startup shows up here as a non-zero exit code plus its own
    # stderr
    print("\nBoot check (each server should stay running, not exit):\n")
    import subprocess
    import time


    for name, spec in SERVER_SPECS.items():
        try:
            proc = subprocess.Popen(
                [spec["command"], *spec["args"]],
                cwd=spec.get("cwd"),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as exc:
            print(f"  FAIL {name}: could not launch -- {exc}")
            continue


        # Keep stdin OPEN and just watch. Closing it (as communicate() does)
        # sends EOF
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline and proc.poll() is None:
            time.sleep(0.2)


        if proc.poll() is None:
            print(f"  OK   {name}: still running after 6s (healthy)")
            proc.kill()
            proc.communicate()
            continue


        _, err = proc.communicate()
        print(f"  FAIL {name}: exited with code {proc.returncode} instead of "
              f"waiting for input")
        if err.strip():
            for line in err.strip().splitlines()[-12:]:
                print(f"       | {line}")
        print("       ^ that is the real error; over MCP this same event is "
              "reported only as 'server unavailable'")


    print("\nStarting servers over MCP...\n")
    failures = 0
    try:
        status = run_async(probe_servers())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1


    for name, info in status.items():
        if info["ok"]:
            print(f"  OK   {name}: {', '.join(info['tools'])}")
        else:
            failures += 1
            print(f"  FAIL {name}:\n{info['error']}\n")


    print()
    if failures:
        print(f"{failures} server(s) failed. Run one directly for its full "
              f"traceback -- it should start silently and wait:")
        print(f"    {sys.executable} {SERVER_SPECS['weather']['args'][0]}")
        return 1


    print("All MCP servers started and reported their tools.")
    return 0




if __name__ == "__main__":
    raise SystemExit(_diagnose())