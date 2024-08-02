
from gen_global_dataset import gen_new_cipher
from sm4 import *
import random
import numpy as np
from tqdm import tqdm

class SM4_GenCCDataset:
    def __init__(self, exp=4, num_round=4):
        #print("Enter which round you want to capture [1-32]:")
        #num_round = int(input())
        if num_round<1 or num_round>32:
            print("Invalid round...")
            return -1


        # number of samples
        num_samples = 10**exp
        max_limit_128bit = 2**128

        #  delta parameter set as MSB 1 bit
        delta = 2**127

        # key value (Defaults: Random 128bits)
        key = random.randint(0, max_limit_128bit)
        data = np.zeros((num_samples,32), dtype=np.uint8)
        label = np.zeros((num_samples), dtype=np.uint8)
        for num_sample in tqdm(range(0,num_samples)):

            # label value is randomly generated, otherwise we can use pregenerated label dataset
            y = random.randint(0,1)
            label[num_sample] = y
            if y==1:
                plaintext = random.randint(0, max_limit_128bit)
                ciphertext1, _ = encrypt(plaintext, key, num_round)
                ciphertext2 = gen_new_cipher(c1 =ciphertext1, ind = num_sample, bits =128)
            else:
                ciphertext1 = random.randint(0, max_limit_128bit)
                ciphertext2 = gen_new_cipher(c1 =ciphertext1, ind = num_sample, bits =128)
            a = ciphertext1
            b = ciphertext2
            for i in range(0,16):
                data[num_sample][31-i] = a&255
                data[num_sample][15-i] = b&255
                a = a>>8
                b = b>>8
        data = np.unpackbits(data, axis=1)
        np.save(f"SM4_CCData.npy",data)
        np.save(f"SM4_CCLabel.npy",label)
        print('Dataset Generated')

if __name__ == "__main__":
    SM4_GenCCDataset(exp=6)
