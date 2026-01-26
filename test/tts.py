import random
from abc import ABC, abstractmethod

import httpx
import asyncio


async def fetch(i):
    async with httpx.AsyncClient(timeout=20) as client:
        print(f"Request {i} started")
        r = await client.get("http://localhost:8000/register?persona=azzz")
        print(f"Request {i} done: {r.json()}")


async def main():
    await asyncio.gather(fetch(1), fetch(2), fetch(3))


# asyncio.run(main())


class Group(ABC):
    @abstractmethod
    def peek(self):
        ...


class Group1(Group):

    def peek(self):
        return int(random.gauss(mu=25, sigma=4))


class Group2(Group):

    def peek(self):
        return int(random.gauss(mu=6, sigma=2))


if __name__ == "__main__":
    g1 = Group1()
    g2 = Group2()
    # solution