import os
import tempfile
import time

import subprocess

from manifest import Manifest
from external.plugins.lip_sync.core.avatar import Avatar
from external.plugins.lip_sync.core.models import AvatarWave2LipModel

model = AvatarWave2LipModel()

audio_path = "../files/harvard.wav"

manifest = Manifest()

HLS_FOLDER = "hls_stream"
os.makedirs(HLS_FOLDER, exist_ok=True)


def get_buffer():
    audio_path = "/home/arafeh/PycharmProjects/avatar_rag_back_end/avatar/avatar_only/avatar_dir/harvard.wav"
    avatar = Avatar("lisa_casual_720_pl", model)
    avatar.init()
    return avatar.video_buffer(audio_path)


def start_live_stream():
    # Folder to store the generated segments
    segment_folder = os.path.join(HLS_FOLDER, "segments")
    os.makedirs(segment_folder, exist_ok=True)

    # Command to start FFmpeg and generate HLS stream
    command = [
        "ffmpeg",
        "-f", "concat",  # Using concat protocol
        "-safe", "0",  # Allow absolute paths for file list
        "-i", "pipe:0",  # Input is via stdin (streaming from clips dynamically)
        "-c:v", "libx264",  # Video codec
        "-preset", "ultrafast",  # Encoding speed
        "-hls_time", "1",  # Segment duration (in seconds)
        "-hls_list_size", "50",  # Playlist size (keep last 5 segments)
        "-hls_flags", "delete_segments",  # Remove old segments
        "-f", "hls",  # Output format HLS
        os.path.join(segment_folder, "output.m3u8")  # Output playlist
    ]

    ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    for clip in get_buffer():
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
            tmpfile.close()
            clip.write_videofile(tmpfile.name, codec="libx264")
            with open(tmpfile.name, "rb") as f:
                video_data = f.read()
                try:
                    ffmpeg_process.stdin.write(video_data)  # Write the segment to FFmpeg stdin
                    ffmpeg_process.stdin.flush()  # Ensure data is sent to FFmpeg immediately
                except BrokenPipeError:
                    print("Error: FFmpeg process closed unexpectedly.")
                    break
            os.remove(tmpfile.name)  # Delete temporary file after use
        time.sleep(1)  # Small delay between segments (if necessary)

    # Close stdin and wait for FFmpeg process to finish
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()

    # Optionally, capture any errors from stderr
    stderr = ffmpeg_process.stderr.read()
    if stderr:
        print("FFmpeg Error:", stderr.decode())


start_live_stream()
