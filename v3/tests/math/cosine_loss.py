from v3.math.cosine_loss import get_cosine_loss

for i in range(0, 10+1):
    print(i, get_cosine_loss(i))

print('-'*20)

for i in range(25, 35+1):
    print(i, get_cosine_loss(i))