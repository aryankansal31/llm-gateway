"""Create a gateway API key: uv run python -m scripts.create_key --name <name>"""

import argparse
import asyncio

from app.config import get_settings
from app.db import create_api_key, create_pool


async def main(name: str) -> None:
    pool = await create_pool(get_settings().database_url)
    try:
        raw = await create_api_key(pool, name)
    finally:
        await pool.close()
    print(raw)
    print("Store this key now. It is not shown again.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    asyncio.run(main(parser.parse_args().name))
