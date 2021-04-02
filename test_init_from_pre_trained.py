import torch
from transformers import BertModel

model = BertModel.from_pretrained('bert-base-uncased')

print(len(model.state_dict()))

def check(string, num):
    for char in string.split('.'):
        if char.isdigit() and (int(char)+1) % num == 0:
            # print(char)
            return True

    return False

for key in model.state_dict().keys():
    if check(key, 2):
        print("What we want "+"="*20)
        print(key)
        print("="*32)
    else:
        print(key)
