import enum
from tqdm import tqdm
train_loop = tqdm(range(1,100,2), position=0, ncols=100, leave=False)
for i,x in enumerate(train_loop):
    print(i,x)