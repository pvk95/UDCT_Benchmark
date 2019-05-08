import cycleGAN
import re
import sys
from os import environ as cuda_environment
import os
import numpy as np
import time
import h5py

if __name__ == "__main__":
	time_begin = time.time()
	# List of floats
	sub_value_f = {}
	sub_value_f['lambda_c']    = 10.             # Loss multiplier for cycle
	sub_value_f['lambda_h']    = 1.              # Loss multiplier for histogram
	sub_value_f['dis_noise']   = 0.1             # Std of gauss noise added to Dis
	sub_value_f['syn_noise']   = 0.              # Add gaussian noise to syn images to make non-flat backgrounds 
	sub_value_f['real_noise']  = 0.              # Add gaussian noise to real images to make non-flat backgrounds 
	
	# List of ints
	sub_value_i = {}
	sub_value_i['epoch']       = 200             # Number of epochs to be trained
	sub_value_i['num_iterations']  = 400		 # Number of iterations of gradient upfdate for every epoch (irrespective of batch size)
	sub_value_i['cont_train']  = 0               # If not 0, the training is continued from the epoch that was last saved
	sub_value_i['batch_size']  = 5               # Batch size for training
	sub_value_i['buffer_size'] = 50              # Number of history elements used for Dis
	sub_value_i['save']        = 1               # If not 0, model is saved
	sub_value_i['gpu']         = 0               # Choose the GPU ID (if only CPU training, choose nonexistent number)
	sub_value_i['attention']   = 0               # If not 0, add an attention layer in path: Real -> Fake -> Real
	sub_value_i['verbose']     = 0               # If not 0, some network information is being plotted
	sub_value_i['num_samples'] = 1500			 # Num of samples to train the model
	
	# List of strings
	sub_string = {}
	sub_string['name']         = 'CGAN'       # Name of model (should be unique). Is used to save/load models
	sub_string['dataset']      = 'dataset_neuron.h5'      # ATM: 'nanowire' or 'mitochondria'
	sub_string['architecture'] = 'Res6'          # Network architecture: 'Res6' or 'Res9'
	sub_string['deconv']       = 'transpose'     # Upsampling method: 'transpose' or 'resize'
	sub_string['PatchGAN']     = 'Patch70'       # Choose the Gan type: 'Patch34', 'Patch70', 'Patch142', 'MultiPatch'
	sub_string['mode']         = 'training'      # 'train', 'gen_A', 'gen_B', 'grad_test'
	sub_string['valid_file']    = '../cc/Neuron_annotated_dataset.h5' # Data containing annotated data for predictions
    
	# Create complete dictonary
	var_dict  = sub_string.copy()
	var_dict.update(sub_value_i)
	var_dict.update(sub_value_f)

	#The folder where you want to save the experiment/run
	save_folder = re.search('\=(.*)', sys.argv[1]).group(1)
	save_folder = save_folder + '/'
	if not os.path.exists(save_folder):
		os.makedirs(save_folder)
		with open(save_folder + 'log.txt','w') as log:
			log.write("{} folder created\n".format(save_folder))


	# Update all defined parameters in dictionary
	for arg_i in sys.argv[2:]:
		var   = re.search('(.*)\=', arg_i) # everything before the '='
		g_var = var.group(1)
		if g_var in sub_value_i:
			dtype = 'int'
		elif g_var in sub_value_f:
			dtype = 'float'
		elif g_var in sub_string:
			dtype = 'string'
		else:
			print("Unknown key word: " + g_var)
			print("Write parameters as: <key word>=<value>")
			print("Example: 'python main.py buffer_size=32'")
			print("Possible key words: " + str(var_dict.keys()))
			continue
		
		content   = re.search('\=(.*)',arg_i) # everything after the '='
		g_content = content.group(1)
		if dtype == 'int':
			var_dict[g_var] = int(g_content)
		elif dtype == 'float':
			var_dict[g_var] = float(g_content)
		else:
			var_dict[g_var] = g_content
	if not os.path.isfile(var_dict['dataset']):
		raise ValueError('Dataset does not exist. Specify loation of an existing h5 file.')

	# Restrict usage of GPUs
	cuda_environment["CUDA_VISIBLE_DEVICES"] = str(var_dict['gpu'])
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

	# Get the dataset filename
	with open(save_folder+var_dict['name']+"_params.txt", "w") as myfile:
		for key in sorted(var_dict):
			myfile.write(key + "," + str(var_dict[key]) + "\n")
	
	# Find out, if whole network is needed or only the generators
	gen_only = False
	if 'gen' in var_dict['mode']:
		gen_only = True
	
	# Define the model
	model = cycleGAN.Model(\
		mod_name=var_dict['name'],\
		data_file=var_dict['dataset'],\
		valid_file=var_dict['valid_file'],\
		buffer_size=var_dict['buffer_size'],\
		dis_noise=var_dict['dis_noise'],\
        architecture=var_dict['architecture'],\
		save_folder = save_folder,\
        lambda_c=var_dict['lambda_c'],\
        lambda_h=var_dict['lambda_h'],\
        deconv=var_dict['deconv'],\
        attention=var_dict['attention'],\
        patchgan=var_dict['PatchGAN'],\
        verbose=(var_dict['verbose']!=0),\
        gen_only=gen_only)

	# Plot parameter properties, if applicable
	if var_dict['verbose']:
		# Print the number of parameters
		model.print_count_variables()
		model.print_train_and_not_train_variables()
    	
	# Create a graph file
	model.save_graph()
	
	if var_dict['mode'] == 'grad_test':
		# Test gradients if applicable
		model.print_gradients()
        
	elif var_dict['mode'] == 'training':
		# Train the model
		model.train(batch_size=var_dict['batch_size'],lambda_c=var_dict['lambda_c'],lambda_h=var_dict['lambda_h'],\
					save=bool(var_dict['save']),n_epochs=var_dict['epoch'],syn_noise=var_dict['syn_noise'],\
					real_noise=var_dict['real_noise'],num_samples=var_dict['num_samples'],num_iterations=var_dict['num_iterations'])
            
	elif var_dict['mode'] == 'gen_A':
		model.generator_A(batch_size=var_dict['batch_size'],lambda_c=var_dict['lambda_c'],lambda_h=var_dict['lambda_h'])
        
	elif var_dict['mode'] == 'gen_B':
		model.generator_B(batch_size=var_dict['batch_size'],lambda_c=var_dict['lambda_c'],lambda_h=var_dict['lambda_h'])

	elif var_dict['mode'] == 'data_gen_B':
		pred_data = var_dict['valid_file']
		model.data_gen_B(data_file=pred_data,batch_size=var_dict['batch_size'])
	else:
		sys.exit(1)

	proc_dur = (time.time() - time_begin)/3600
	with open(save_folder + 'log.txt','a+') as log:
		log.write("\nTotal duration: {:.2f}\n".format(proc_dur))

'''
import h5py
import numpy as np

data_file = h5py.File('new_dataset_neuron.h5','r')
images_a = data_file['A/data']

images_a.shape
images_a.dtype

images_ann = images_a[:50,:,:,:]

np.max(images_ann)
images_ann.dtype

np.max(images_a)
np.min(images_a)

images_b = data_file['B/data']

images_b.shape
images_b.dtype

np.max(images_b)
np.min(images_b)

pred_file = h5py.File('../190311_count_ception/Neuron_annotated_dataset.h5','r')
#pred_file = h5py.File('../190311_count_ception/Neuron_annotated_dataset.h5', 'r')
images_ann = pred_file['raw/data'][:50,:,:,:]

images_ann.shape
images_ann.dtype
np.max(images_ann)
np.min(images_ann)

images_ann = (np.minimum(np.maximum(images_ann, 0), 1) * (2 ** 16 - 1)).astype(np.uint16)

images_ann.shape
np.max(images_ann)
images_new = np.concatenate((images_ann,images_a),axis=0)

data_file_new = h5py.File(newFileName,'w')
group = data_file_new.create_group('A')
group.create_dataset(name='data', data=images_new, dtype=np.uint16)
group = data_file_new.create_group('B')
group.create_dataset(name='data', data=data_file['B/data'], dtype=np.uint16)

data_file_new.close()
pred_file.close()
data_file.close()

'''
