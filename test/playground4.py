from core.plugins.lip_sync.core.avatar import AvatarManager
from core.plugins.lip_sync.core.models import AvatarWave2LipModel
from core.plugins.text2speech import MicrosoftText2Speech
from manifest import Manifest

text = ("Flare is an innovative blockchain network that brings smart contract functionality and interoperability to a "
        "diverse array of decentralized applications. By integrating with existing blockchain ecosystems, "
        "Flare enables the seamless exchange of data and value across different networks, paving the way for enhanced "
        "liquidity, robust decentralized finance (DeFi) solutions, and broader adoption of digital assets. Its unique "
        "approach to combining the benefits of traditional blockchains with advanced features such as autonomous "
        "token distribution, like FlareDrops, positions it as a pivotal player in the evolution of blockchain "
        "technology. Through initiatives like FlareDollar (USDF) and ongoing community engagement, the network "
        "continues to innovate, aiming to drive both technological progress and real-world utility in the rapidly "
        "expanding crypto landscape.")

avatar_manager = AvatarManager(AvatarWave2LipModel(), MicrosoftText2Speech())

buffer = avatar_manager.tts_buffer('lisa_casual_720_pl',
                                   text,
                                   voice_id=7406)


def count_gen(gen):
    j = 0
    for _ in gen:
        j += 1
    return j


for avatar_stream, audio in buffer:
    print(audio, count_gen(avatar_stream))
