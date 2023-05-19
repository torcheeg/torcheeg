def get_data():
    for i in range(10):
        yield i

while True:
    try:
        print(next(get_data()))
    except StopIteration:
        break