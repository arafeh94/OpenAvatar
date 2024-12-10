from apis.test2speech import Client
from apis.test2speech.api.speech import generate_speech
from apis.test2speech.api.speech import generate_speech_post
from apis.test2speech.models import SpeechRequest

client = Client('http://127.0.0.1:8000')
text = 'what you know about walking down in the deep, people keep dream on but nothing get to sleep.'

# use get
response = generate_speech.sync_detailed(client=client, text=text, voice_id=7306)
open('test1.wav', 'wb').write(response.content)

speech_request = SpeechRequest(text="post:" + text)
response = generate_speech_post.sync_detailed(client=client, voice_id=7306, body=speech_request)
open('test2.wav', 'wb').write(response.content)
# use post
