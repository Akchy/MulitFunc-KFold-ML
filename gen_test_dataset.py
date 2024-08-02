import random
import gen_global_dataset
from sm4 import encrypt
from tqdm import tqdm

def gen_test_values(num, file_name):
    max_limit_128bit = 2**128
    key = random.randint(0, max_limit_128bit)
    data = []
    label = []
    for num_sample in tqdm(range(0,num)):

        # label value is randomly generated, otherwise we can use pregenerated label dataset
        y = random.randint(0,1)
        if y==1:
            plaintext = random.randint(0, max_limit_128bit)
            ciphertext1, _ = encrypt(plaintext, key, 4)
            ciphertext2 = gen_global_dataset.gen_new_cipher(c1 =ciphertext1, ind = num_sample, bits =128)
        else:
            ciphertext1 = random.randint(0, max_limit_128bit)
            ciphertext2 = gen_global_dataset.gen_new_cipher(c1 =ciphertext1, ind = num_sample, bits =128)
        
        cipher = bin(ciphertext1)[2:].zfill(128) + bin(ciphertext2)[2:].zfill(128)
        data.append(cipher)
        label.append(y)

    file = open(file_name, 'w+')
    file1 = open('lab.txt', 'w+')
    for i in range(num):
        file.write(f'{data[i]}\n')
        file1.write(f'{label[i]}\n')
    print('Test Data Generated')
