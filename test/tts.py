import httpx
import asyncio

async def fetch(i):
    async with httpx.AsyncClient(timeout=20) as client:
        print(f"Request {i} started")
        r = await client.get("http://localhost:8000/register?persona=azzz")
        print(f"Request {i} done: {r.json()}")

async def main():
    await asyncio.gather(fetch(1), fetch(2), fetch(3))

asyncio.run(main())