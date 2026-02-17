import pycurl
import os
from io import BytesIO

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
AUDIO_PATH = os.path.expanduser("./consent.mp3")

buffer = BytesIO()
c = pycurl.Curl()

c.setopt(c.URL, "https://api.openai.com/v1/audio/voice_consents")
c.setopt(c.WRITEDATA, buffer)

c.setopt(
    c.HTTPHEADER,
    [
        f"Authorization: Bearer {OPENAI_API_KEY}",
    ],
)

form = [
    ("name", "test_consent"),
    ("language", "en"),
    (
        "recording",
        (
            c.FORM_FILE,
            AUDIO_PATH,
            c.FORM_CONTENTTYPE,
            "audio/x-wav",
        ),
    ),
]

c.setopt(c.HTTPPOST, form)

c.perform()
c.close()

response = buffer.getvalue().decode("utf-8")
print(response)
