import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np

from utils import utils, helpers
from builders import model_builder


image='/Users/kaygudo/workspace/ETH/Perception and Learning for Robotics/rough_work/sony/DSC04802.JPG'#The image you want to predict on.
checkpoint_path= '/Users/kaygudo/workspace/ETH/Perception and Learning for Robotics/rough_work/cluster_train_result1/results_batch2_newclass_dataaug/checkpoints/latest_model_FRRN-A_tugraz.ckpt.data-00000-of-00001 '  #The path to the latest checkpoint weights for your model.
crop_height= 512
crop_width=512
model='FRRN-A' #The model you are using
dataset="tugraz"

class_names_list, label_values = helpers.get_label_info(os.path.join(dataset, "class_dict.csv"))

num_classes = len(label_values)

print("\n***** Begin prediction *****")
print("Dataset -->", dataset)
print("Model -->", model)
print("Crop Height -->", crop_height)
print("Crop Width -->", crop_width)
print("Num Classes -->", num_classes)
print("Image -->", image)

# Initializing network
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes]) 

network, _ = model_builder.build_model(model, net_input=net_input,
                                        num_classes=num_classes,
                                        crop_width=crop_width,
                                        crop_height=crop_height,
                                        is_training=False)

sess.run(tf.global_variables_initializer())

print('Loading model checkpoint weights')
saver=tf.train.Saver(max_to_keep=1000)
saver.restore(sess, checkpoint_path)


print("Testing image " + image)

loaded_image = utils.load_image(image)
resized_image =cv2.resize(loaded_image, (crop_width, crop_height))
input_image = np.expand_dims(np.float32(resized_image[:crop_height, :crop_width]),axis=0)/255.0

st = time.time()
output_image = sess.run(network,feed_dict={net_input:input_image})

run_time = time.time()-st

output_image = np.array(output_image[0,:,:,:])
output_image = helpers.reverse_one_hot(output_image)

out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
file_name = utils.filepath_to_name(image)
cv2.imwrite("%s_pred.png"%(file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))

print("")
print("Finished!")
print("Wrote image " + "%s_pred.png"%(file_name))
