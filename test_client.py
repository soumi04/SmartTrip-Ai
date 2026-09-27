import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():

    # ========================================================
    # CONNECT TO MCP SERVER
    # ========================================================

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).resolve().parent / "mcp_server" / "server.py")],
        cwd=Path(__file__).resolve().parent,
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as client:
            await client.initialize()

            # ====================================================
            # LIST AVAILABLE MCP TOOLS
            # ====================================================

            result = await client.list_tools()

            print("\n========================================")
            print("       SMARTTRIP MCP SERVER")
            print("========================================")

            print("\nAvailable MCP tools:")

            for tool in result.tools:
                print("-", tool.name)

            # ====================================================
            # TOOL 1: SEARCH FLIGHTS
            # ====================================================

            print("\n========================================")
            print("          FLIGHT SEARCH")
            print("========================================")

            flight_result = await client.call_tool(
                "search_flights",
                {
                    "origin": "Chennai",
                    "destination": "Singapore",
                    "date": "2026-12-10"
                }
            )

            print(flight_result)

            # ====================================================
            # TOOL 2: SEARCH HOTELS
            # ====================================================

            print("\n========================================")
            print("          HOTEL SEARCH")
            print("========================================")

            hotel_result = await client.call_tool(
                "search_hotels",
                {
                    "city": "Singapore"
                }
            )

            print(hotel_result)

            # ====================================================
            # TOOL 3: CONVERT CURRENCY
            # ====================================================

            print("\n========================================")
            print("       CURRENCY CONVERSION")
            print("========================================")

            currency_result = await client.call_tool(
                "convert_currency",
                {
                    "amount": 80000,
                    "from_currency": "INR",
                    "to_currency": "SGD"
                }
            )

            print(currency_result)


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())