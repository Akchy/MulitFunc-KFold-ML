# temp.py
import random
from tqdm import tqdm

def rightRotate(n, d, INT_BITS):
    return (n >> d) |(n << (INT_BITS - d)) & (2**INT_BITS - 1)

def rotate_func(bits, no_of_bits):
    # Rotate Bits at position: 1,3,5,7 (either one)
    rotate_bits_arr = [1,3,5,7]
    rotate_ind = random.randint(0,len(rotate_bits_arr)-1)
    rotate_val = rotate_bits_arr[rotate_ind]
    rotated_bits = rightRotate(bits, rotate_val , no_of_bits)
    return rotated_bits

def flip_bits_func(bits):
    # Flip Bits at position: 4, 8, 12 (either one) 
    flip_bits_arr = [4, 8, 12]
    flip_ind = random.randint(0,len(flip_bits_arr)-1)
    flip_bit_val = flip_bits_arr[flip_ind]
    return bits ^ (2**(flip_bit_val-1))

def flip_byte_func(bits):
    # Flip Bytes at position: 1, 2, 4 (either one) 
    alpha = random.randint(1, 255)
    bit_str = bin(bits)[2:]
    bit_str = bit_str.zfill(32)
    byte_flip_arr = [1, 2, 4]
    
    reverse_bit = bit_str[::-1]
    flip_byte_ind = random.randint(0,len(byte_flip_arr)-1)
    flip_byte_val = byte_flip_arr[flip_byte_ind]
    # print('which byte: ',flip_byte_val)
    
    start_ind = (flip_byte_val-1)*8
    # print('start ind: ',start_ind)
    end_ind = ((flip_byte_val-1)*8)+8
    
    middle_bits = reverse_bit[start_ind:end_ind]
    # print('bit: ',reverse_bit)
    # print('old middle: ',middle_bits)
    middle_int = int(middle_bits,2)
    
    xor = middle_int ^ alpha
    new_middle = str(bin(xor)[2:]).zfill(8)
    # print('new middle: ',new_middle)
    
    starting_bits = reverse_bit[:start_ind]
    end_bits = reverse_bit[end_ind:]
    new_bit = starting_bits + new_middle + end_bits
    # print('new bit: ',new_bit)
    return int(new_bit[::-1],2)

def gen_new_cipher(c1,ind, bits):
    diff_func = [rotate_func,flip_bits_func,flip_byte_func]
    func_ind = ind%len(diff_func)
    func = diff_func[func_ind]
    if(func_ind == 0):
        c2 = func(c1, bits)
    else:
        c2 = func(c1)
    return c2


if __name__ == '__main__':
    samples = 10
    for i in tqdm(range(samples)):

        no_of_bits = 128
        max_limit_128bit = 2**no_of_bits
        ciphertext1 = random.randint(0, max_limit_128bit)
        diff_func = [rotate_func,flip_bits_func,flip_byte_func]
        func_ind = i%len(diff_func)
        func = diff_func[func_ind]
        if(func_ind == 0):
            ciphertext2 = func(ciphertext1, no_of_bits)
        else:
            ciphertext2 = func(ciphertext1)
        print(str(bin(ciphertext1)[2:])+ str(bin(ciphertext2)[2:]))