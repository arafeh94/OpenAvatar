import sounddevice as sd
import librosa

from core.plugins.text2speech import MicrosoftText2Speech, Audio


def ror():
    sr = ['ahmad']
    print('0')
    for i in range(3):
        print('i')
        sr.append(str(i))
        for ii in range(3):
            print('ii')
            sr.append('i' + str(ii))
            yield i


dg = ror()
dg.__next__()
dg.__next__()
dg.__next__()
dg.__next__()
dg.__next__()
dg.__next__()
dg.__next__()
dg.__next__()
