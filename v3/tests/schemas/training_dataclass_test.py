from v3.data.get_workouts import Workouts


if __name__ == '__main__':
    w = Workouts()
    w.generate()
    x, y = w.to_tensor()
    print(x)
    print(x.shape)
    print(y)
    print(y.shape)