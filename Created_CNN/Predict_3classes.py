import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
import xlsxwriter

classes = os.listdir('training_data')

print(classes)

##### IMPORTANT ########

# change name of Exel book (l.18), and data model (l.68) to make each try traceable!!!!!
# _1 done,_2 done, _3 done



wb=xlsxwriter.Workbook('Conf_Matrix_back_test.xlsx')
ws=wb.add_worksheet()

ws.write(0, 1, 'Data_Wallet')
ws.write(0, 2, 'Data_Keys')
ws.write(0, 3, 'Data_Background')

ws.write(1, 0, 'Result_Wallet')
ws.write(2,0, 'Result_Keys')
ws.write(3,0, 'Result_Background')

ws.write(4,0,'class_data_W')
ws.write(4,1,'accuracy')
ws.write(4,2,'class_data_K')
ws.write(4,3,'accuracy')
ws.write(4,4,'class_data_B')
ws.write(4,5,'accuracy')





for clase in classes:
	
	dir_path= 'C:/Users/Jesus/Desktop/CNN_building/image-classifier/testing_data/'+clase
	n=1
	contador_k=0
	contador_w=0
	contador_b=0
	r_acc=5

	if clase=='wallet':
		col=1
		c_clas=0
		col_acc=1
	elif clase=='keys':
		col=2
		c_clas=2
		col_acc=3
	elif clase=='background':
		col=3
		c_clas=4
		col_acc=5
  
	

	for filename in glob.glob(dir_path +'/*.jpg'):

		image_name=filename.replace(dir_path,'')

		image_size=128
		num_channels=3
		images = []
		# Reading the image using OpenCV
		image = cv2.imread(filename)
		# Resizing the image to our desired size and preprocessing will be done exactly as done during training
		image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
		images.append(image)
		images = np.array(images, dtype=np.uint8)
		images = images.astype('float32')
		images = np.multiply(images, 1.0/255.0) 
		#The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
		x_batch = images.reshape(1, image_size,image_size,num_channels)

		## Let us restore the saved model 
		sess = tf.Session()
		# Step-1: Recreate the network graph. At this step only graph is created.
		saver = tf.train.import_meta_graph('keys-wallet-back-LR-test.meta')
		# Step-2: Now let's load the weights saved using the restore method.
		saver.restore(sess, tf.train.latest_checkpoint('./'))

		# Accessing the default graph which we have restored
		graph = tf.get_default_graph()

		# Now, let's get hold of the op that we can be processed to get the output.
		# In the original network y_pred is the tensor that is the prediction of the network
		y_pred = graph.get_tensor_by_name("y_pred:0")

		## Let's feed the images to the input placeholders
		x= graph.get_tensor_by_name("x:0") 
		y_true = graph.get_tensor_by_name("y_true:0") 
		y_test_images = np.zeros((1, len(os.listdir('training_data')))) 


		### Creating the feed_dict that is required to be fed to calculate y_pred 
		feed_dict_testing = {x: x_batch, y_true: y_test_images}
		result=sess.run(y_pred, feed_dict=feed_dict_testing)



		r1=str(result[0,0]*100) #background
		r2=str(result[0,1]*100) #keys
		r3=str(result[0,2]*100) #wallet

		rs=(result[0,0]*10**12,result[0,1]*10**12,result[0,2]*10**12)
		max_=max(rs)
		max_class=rs.index(max_)
		

		if max_class==1:
			label_r='keys'
			row=2
			contador_k=contador_k+1
			ws.write(row, col, contador_k)

		elif max_class==2:
			label_r='wallet'
			row=1
			contador_w=contador_w+1
			ws.write(row, col, contador_w)
		elif max_class==0:
			label_r='background'
			row=3
			contador_b=contador_b+1
			ws.write(row, col, contador_b)


		ws.write(r_acc,col_acc,rs[max_class]/(10**12))
		ws.write(r_acc,c_clas,label_r)
		r_acc=r_acc+1
		# result is of this format [probabiliy_of_keys probability_of_wallet]
		output=(classes[0]+':'+r1[0:5]+'%'+'\n'+classes[1]+':'+r2[0:5]+'%'+'\n'+classes[2]+':'+r3[0:5]+'%')
		print('Results for image {}>>\n{}'.format(image_name,output))

wb.close()