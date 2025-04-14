import av


container = av.open('avatar.mp4')
gen = container.decode(*container.streams)
for i in gen:
    print(i)
