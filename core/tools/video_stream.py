from typing import Generator

from fastapi.responses import StreamingResponse


def video_stream_generator(file_path: str) -> Generator[bytes, None, None]:
    with open(file_path, "rb") as video_file:
        while chunk := video_file.read(1024 * 1024):
            yield chunk


def video_stream(file_path: str, media_type: str = "video/mp4"):
    return StreamingResponse(video_stream_generator(file_path), media_type=media_type)
