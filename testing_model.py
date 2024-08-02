import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from keras.models import load_model
from keras.layers import Dropout, Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from keras.regularizers import l2
from keras.models import Model, load_model
from gen_test_dataset import * 
from sklearn.metrics import accuracy_score


#ciphername no round, model (1dcnn)
class TestingModel:
	
	def getResult(self, model, testing_file, cipher_name,label_file_name):
		
		# res = []
		y_pred = []
		with open(testing_file, 'r') as file:
			for num, line in tqdm(enumerate(file)):
				list1 = [int(x) for x in line.strip()]
				cipher = [list1]
				y = [1,]
				output_str = ''
				# print(f'{cipher}, {y}, {type(cipher)}, {type(y)}')
				result = model.predict(cipher,verbose = None)
				pred = result[0][0]
				#if pred >= 0.5:
				# output_str = f'The ciphertext at line {num+1} belongs to {cipher_name} as per selected Model with {round(pred)}% accuracy'
				y_pred.append(round(pred))
				# res.append(output_str)
		# for i in res:
		# 	print(i)
		y_test = []
		with open(label_file_name, 'r')as label_file:
			for line in label_file:
				y_test.append(int(line))
		return y_test, y_pred


#make residual tower of convolutional blocks
def make_resnet(num_blocks=1, num_filters=32, num_outputs=1, d1=512, d2=512, word_size=128, ks=3, depth=5, reg_param=0.0001, final_activation='sigmoid'):
  #Input and preprocessing layers
  inp = Input(shape=(num_blocks * word_size * 2,))
  rs = Reshape((2 * num_blocks, word_size))(inp)
  perm = Permute((2,1))(rs)
  #add a single residual layer that will expand the data to num_filters channels
  #this is a bit-sliced layer
  conv0 = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(perm)
  conv0 = BatchNormalization()(conv0)
  conv0 = Activation('relu')(conv0)
  #add residual blocks
  shortcut = conv0
  for i in range(depth):
    conv1 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv1D(num_filters, kernel_size=ks, padding='same',kernel_regularizer=l2(reg_param))(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    shortcut = Add()([shortcut, conv2])
    
    
  #add prediction head
  flat1 = Flatten()(shortcut)
  dense1 = Dense(d1,kernel_regularizer=l2(reg_param))(flat1)#output array of shape d1and regularizer
  dense1 = BatchNormalization()(dense1)
  dense1 = Activation('relu')(dense1)
  dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
  dense2 = BatchNormalization()(dense2)
  dense2 = Activation('relu')(dense2)
  out = Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param))(dense2)
  model = Model(inputs=inp, outputs=out)#creates model using the specified inputs and output layer
  
  return(model)


if __name__ == "__main__":
	tm = TestingModel()
	net = make_resnet(depth=1, reg_param=10**-5)
	filename = 'vals1.txt'
	label_file_name = 'lab.txt'
	modelname = 'sm4_c1c2_ccn1d.h5'
	gen_test_values(1000, filename)
	net.load_weights(modelname)
	(y_test, y_pred) = tm.getResult(net,filename,'SM4', label_file_name)
	accuracy = accuracy_score(y_test , y_pred)
	print(accuracy)
