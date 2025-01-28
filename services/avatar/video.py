import pickle


def as_buffer(batches):
    for batch in batches:
        yield batch


frames_batch = pickle.load(open("../../files/harvard.pkl", "rb"))
buffer = as_buffer(frames_batch)
print(1, next(buffer))
print(2, next(buffer))
print(3, next(buffer))
print(4, next(buffer))
print(5, next(buffer))
print(6, next(buffer))
print(next(buffer))
print(next(buffer))
print(next(buffer))
print(next(buffer))
