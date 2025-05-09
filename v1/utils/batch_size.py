def calc_batch_size(total_samples):
    estimate = int(total_samples ** 0.5) + 1
    last_batch = total_samples % estimate

    min_last_batch = estimate - 1

    while 0 < last_batch < min_last_batch:
        estimate += 1
        last_batch = total_samples % estimate

    return estimate


for i in range(12, 30+1):
    batch_size = calc_batch_size(i)
    print(i, batch_size, i % batch_size)