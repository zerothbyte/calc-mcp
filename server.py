import math
import argparse
from mcp.server.fastmcp import FastMCP
from tools import register_all

mcp = FastMCP("Calculator")
register_all(mcp)


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


@mcp.tool()
def power(base: float, exponent: float) -> float:
    """Raise base to the power of exponent."""
    return math.pow(base, exponent)


@mcp.tool()
def sqrt(a: float) -> float:
    """Calculate the square root of a number."""
    if a < 0:
        raise ValueError("Cannot calculate square root of a negative number")
    return math.sqrt(a)


@mcp.tool()
def modulo(a: float, b: float) -> float:
    """Calculate the remainder of a divided by b."""
    if b == 0:
        raise ValueError("Cannot modulo by zero")
    return a % b


@mcp.tool()
def log(a: float, base: float = math.e) -> float:
    """Calculate logarithm of a with given base (default: natural log)."""
    if a <= 0:
        raise ValueError("Logarithm undefined for non-positive numbers")
    if base <= 0 or base == 1:
        raise ValueError("Log base must be positive and not equal to 1")
    return math.log(a, base)


@mcp.tool()
def sin(angle_degrees: float) -> float:
    """Calculate sine of an angle in degrees."""
    return math.sin(math.radians(angle_degrees))


@mcp.tool()
def cos(angle_degrees: float) -> float:
    """Calculate cosine of an angle in degrees."""
    return math.cos(math.radians(angle_degrees))


@mcp.tool()
def tan(angle_degrees: float) -> float:
    """Calculate tangent of an angle in degrees."""
    return math.tan(math.radians(angle_degrees))


@mcp.tool()
def factorial(n: int) -> int:
    """Calculate factorial of a non-negative integer."""
    if n < 0:
        raise ValueError("Factorial undefined for negative numbers")
    return math.factorial(n)


@mcp.tool()
def absolute(a: float) -> float:
    """Calculate the absolute value of a number."""
    return abs(a)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculator MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http", "all"],
        default="stdio",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    mcp.settings.host = args.host
    mcp.settings.port = args.port

    if args.transport == "all":
        import uvicorn
        from contextlib import asynccontextmanager
        from starlette.applications import Starlette
        from starlette.routing import Route, Mount
        from starlette.responses import Response
        from mcp.server.sse import SseServerTransport
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        from mcp.server.fastmcp.server import StreamableHTTPASGIApp

        # SSE transport
        sse = SseServerTransport("/calc/messages/")

        # Streamable HTTP transport
        session_manager = StreamableHTTPSessionManager(
            app=mcp._mcp_server,
            json_response=mcp.settings.json_response,
            stateless=mcp.settings.stateless_http,
        )
        http_handler = StreamableHTTPASGIApp(session_manager)

        async def handle_sse(request):
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await mcp._mcp_server.run(
                    streams[0], streams[1], mcp._mcp_server.create_initialization_options()
                )
            return Response(status_code=200)

        async def handle_sse_post(request):
            await sse.handle_post_message(request.scope, request.receive, request._send)

        @asynccontextmanager
        async def lifespan(app):
            async with session_manager.run():
                yield

        app = Starlette(
            routes=[
                Route("/calc/sse", endpoint=handle_sse),
                Route("/calc/messages/", endpoint=handle_sse_post, methods=["POST"]),
                Route("/calc/mcp", endpoint=http_handler, methods=["GET", "POST", "DELETE"]),
            ],
            lifespan=lifespan,
        )

        uvicorn.run(app, host=args.host, port=args.port)
    else:
        mcp.run(transport=args.transport)
