import asyncio

from services.avatar.service import request_avatar, get_next_clip

text = "Hello, this is a test text."
tokens_dict = asyncio.run(request_avatar(text, 1))
token = tokens_dict['token']
while True:
    clip = get_next_clip(token)
    if clip is None:
        break
    clip.preview()
