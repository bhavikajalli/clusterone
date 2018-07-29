
# coding: utf-8

# In[ ]:


from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import tensorflow as tf
from PIL import Image
import os
import tensorflow as tf
from clusterone import get_data_path, get_logs_path


# In[ ]:


get_ipython().magic('matplotlib inline')
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144


# In[ ]:


train_dir = get_data_path(
            dataset_name = 'data/bhavikaj/bhavikajalli-clusterone',  # on ClusterOne
            local_root = 'asl_alphabet_train/',  # path to local dataset
            local_repo = '',  # local data folder name
            path = ''  # folder within the data folder
            )


# In[ ]:


#train_dir = "asl_alphabet_train/"
#train_dir = "ex/"
#train_dir = "data/bhavikaj/bhavikajalli-clusterone"

imageSize = 50
def get_data(folder):
    """
    Load the data and labels from the given folder.
    """
    X = []
    y = []
    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            if folderName in ['A']:
                label = 0
            elif folderName in ['B']:
                label = 1
            elif folderName in ['C']:
                label = 2
            elif folderName in ['D']:
                label = 3
            elif folderName in ['E']:
                label = 4
            elif folderName in ['F']:
                label = 5
            elif folderName in ['G']:
                label = 6
            elif folderName in ['H']:
                label = 7
            elif folderName in ['I']:
                label = 8
            elif folderName in ['J']:
                label = 9
            elif folderName in ['K']:
                label = 10
            elif folderName in ['L']:
                label = 11
            elif folderName in ['M']:
                label = 12
            elif folderName in ['N']:
                label = 13
            elif folderName in ['O']:
                label = 14
            elif folderName in ['P']:
                label = 15
            elif folderName in ['Q']:
                label = 16
            elif folderName in ['R']:
                label = 17
            elif folderName in ['S']:
                label = 18
            elif folderName in ['T']:
                label = 19
            elif folderName in ['U']:
                label = 20
            elif folderName in ['V']:
                label = 21
            elif folderName in ['W']:
                label = 22
            elif folderName in ['X']:
                label = 23
            elif folderName in ['Y']:
                label = 24
            elif folderName in ['Z']:
                label = 25
            elif folderName in ['del']:
                label = 26
            elif folderName in ['nothing']:
                label = 27
            elif folderName in ['space']:
                label = 28           
            else:
                label = 29
            for image_filename in tqdm(os.listdir(folder + folderName)):
                img_file = np.asarray(Image.open(folder + folderName + '/' + image_filename)).astype(np.float32)/255.0
                #Image.open('img1.png')
                if img_file is not None:
                    img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3))
                    img_arr = np.asarray(img_file).astype(np.float32)
                    X.append(img_arr)
                    y.append(label)
#             i += 1
#             if i == 100:
#                 break
    X = np.asarray(X)
    y = np.asarray(y)
    
    return X,y
X_train, y_train = get_data(train_dir) 
#X_test, y_test= get_data(test_dir) # Too few images


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2) 

# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

y_trainHot = to_categorical(y_train, num_classes = 30)
y_testHot = to_categorical(y_test, num_classes = 30)


# In[ ]:


len(y_trainHot[0])
print(X_train[0].dtype)
print(y_trainHot.dtype)
print(X_test[0].dtype)


# In[ ]:


import matplotlib.pyplot as plt

#Visualize the dataset
def plotHistogram(a):
    """
    Plot histogram of RGB Pixel Intensities
    """
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(a)
    plt.axis('off')
    histo = plt.subplot(1,2,2)
    histo.set_ylabel('Count')
    histo.set_xlabel('Pixel Intensity')
    n_bins = 30
    plt.hist(a[:,:,0].flatten(), bins= n_bins, lw = 0, color='r', alpha=0.5);
    plt.hist(a[:,:,1].flatten(), bins= n_bins, lw = 0, color='g', alpha=0.5);
    plt.hist(a[:,:,2].flatten(), bins= n_bins, lw = 0, color='b', alpha=0.5);
#plotHistogram(X_train[1])


# In[ ]:


##Defining the Model
#Training using a simple model
learning_rate = 0.001
num_steps = 2000
training_epochs = 1
batch_size = 128
display_step = 100

# Network Parameters
n_input = imageSize # ASL data input (img shape: 200*200*3)
n_classes = 30 # ASL total classes (0-25 alphabets, 1 space,del,nothing and other digits)
dropout = 0.5



# In[ ]:


#Defining and building the graph
def conv_net(x, n_classes, dropout,is_training):
    # Define a scope for reusing the variables
    
    #The ASL data input is a 3-D vector of 200x200 features
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 200, 200, 3])

    # Convolution Layer with 64 filters and a kernel size of 5
    x = tf.layers.conv2d(x, 64, 5, activation=tf.nn.relu)
    # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
    x = tf.layers.max_pooling2d(x, 2, 2)

    # Convolution Layer with 256 filters and a kernel size of 5
    x = tf.layers.conv2d(x, 128, 3, activation=tf.nn.relu)
    # Convolution Layer with 512 filters and a kernel size of 5
    x = tf.layers.conv2d(x, 256, 3, activation=tf.nn.relu)
    # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
    x = tf.layers.max_pooling2d(x, 2, 2)

    # Flatten the data to a 1-D vector for the fully connected layer
    x = tf.contrib.layers.flatten(x)

    #x = tf.layers.dense(x, 2048)
    # Apply Dropout (if is_training is False, dropout is not applied)
    #x = tf.layers.dropout(x, rate=dropout, training=is_training)

    # Fully connected layer (in contrib folder for now)
    x = tf.layers.dense(x, 1024)
    # Apply Dropout (if is_training is False, dropout is not applied)
    x = tf.layers.dropout(x, rate=dropout, training=is_training)

    # Output layer, class prediction
    out = tf.layers.dense(x, n_classes)
    # Because 'softmax_cross_entropy_with_logits' loss already apply
    # softmax, we only apply softmax to testing network
    out = tf.nn.softmax(out) if not is_training else out

    return out


# In[ ]:


def next_batch(batch_size, data, labels):
    '''
    Return a total of `batch_size` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


# In[ ]:


#Create logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

##Defining the Local variables
CLUSTERONE_USERNAME = "bhavikaj"

# Where should your local log files be stored? This should be something like "~/Documents/self-driving-demo/logs/"
LOCAL_LOG_LOCATION = "~/Documents/Github/clusterone/logs/"

# Where is the dataset located? This should be something like "~/Documents/data/" if the dataset is in "~/Documents/data/comma"
LOCAL_DATASET_LOCATION = "~/Documents/Github/clusterone/asl_alphabet_train/"

# Name of the data folder. In the example above, "comma"
LOCAL_DATASET_NAME = "asl_alphabet_train"

train_data_dir = "~/Documents/Github/clusterone/asl_alphabet_train/"


# In[ ]:


##Graph Creation and Training for the distributed training platform
try:
    job_name = os.environ['JOB_NAME']
    task_index = os.environ['TASK_INDEX']
    ps_hosts = os.environ['PS_HOSTS']
    worker_hosts = os.environ['WORKER_HOSTS']
except: # we are not on TensorPort, assuming local, single node
    job_name = None
    task_index = 0
    ps_hosts = None
    worker_hosts = None
    
if job_name == None: #if running locally
    if LOCAL_LOG_LOCATION == "...":
        raise ValueError("LOCAL_LOG_LOCATION needs to be defined")
    if LOCAL_DATASET_LOCATION == "...":
        raise ValueError("LOCAL_DATASET_LOCATION needs to be defined")
    if LOCAL_DATASET_NAME == "...":
        raise ValueError("LOCAL_DATASET_NAME needs to be defined")
        


# In[ ]:


##To enable running locally and on clusterone
PATH_TO_LOCAL_LOGS = os.path.expanduser(LOCAL_LOG_LOCATION)
ROOT_PATH_TO_LOCAL_DATA = os.path.expanduser(LOCAL_DATASET_LOCATION)


# In[ ]:


#Flags
import sys

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("logs_dir",
    get_logs_path(root=PATH_TO_LOCAL_LOGS),
    "Path to store logs and checkpoints. It is recommended"
    "to use get_logs_path() to define your logs directory."
    "If you set your logs directory manually make sure"
    "to use /logs/ when running on TensorPort cloud.")

flags.DEFINE_string("job_name", job_name,
                        "job name: worker or ps")
flags.DEFINE_integer("task_index", task_index,
                    "Worker task index, should be >= 0. task_index=0 is "
                    "the chief worker task the performs the variable "
                    "initialization")
flags.DEFINE_string("ps_hosts", ps_hosts,
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", worker_hosts,
                    "Comma-separated list of hostname:port pairs")

remaining_args = FLAGS([sys.argv[0]] + [flag for flag in sys.argv if flag.startswith("--")])
assert(remaining_args == [sys.argv[0]])
FLAGS = flags.FLAGS


flags.DEFINE_integer("steps_per_epoch", 10, "Number of training steps per epoch")
flags.DEFINE_integer("nb_epochs", 20, "Number of epochs")


# In[ ]:


# This function defines the master, ClusterSpecs and device setters
def device_and_target():
    # If FLAGS.job_name is not set, we're running single-machine TensorFlow.
    # Don't set a device.
    if FLAGS.job_name is None:
        print("Running single-machine training")
        return (None, "")

    # Otherwise we're running distributed TensorFlow.
    print("Running distributed training")
    if FLAGS.task_index is None or FLAGS.task_index == "":
        raise ValueError("Must specify an explicit `task_index`")
    if FLAGS.ps_hosts is None or FLAGS.ps_hosts == "":
        raise ValueError("Must specify an explicit `ps_hosts`")
    if FLAGS.worker_hosts is None or FLAGS.worker_hosts == "":
        raise ValueError("Must specify an explicit `worker_hosts`")

    cluster_spec = tf.train.ClusterSpec({
            "ps": FLAGS.ps_hosts.split(","),
            "worker": FLAGS.worker_hosts.split(","),
    })
    server = tf.train.Server(
            cluster_spec, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
        server.join()

    worker_device = "/job:worker/task:{}".format(FLAGS.task_index)
    # The device setter will automatically place Variables ops on separate
    # parameter servers (ps). The non-Variable ops will be placed on the workers.
    return (
            tf.train.replica_device_setter(
                    worker_device=worker_device,
                    cluster=cluster_spec),
            server.target,
    )

device, target = device_and_target()        


# In[ ]:


print(FLAGS.logs_dir)
if FLAGS.logs_dir is None or FLAGS.logs_dir == "":
        raise ValueError("Must specify an explicit `logs_dir`")


# In[ ]:


# Defining graph
# # tf Graph input

reuse_vars = False
tf.reset_default_graph()
with tf.device(device):
    
        #TODO define your graph here
    #batch_x, batch_y = next_batch(batch_size,X_train,y_trainHot)
    #x = batch_x
    #print(x.shape)
    #y = batch_y
    x = tf.placeholder("float32", [None, n_input,n_input,3])
    y = tf.placeholder("float32", [None, n_classes])
    logits = conv_net(x,n_classes, dropout,is_training=True)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits, labels=y))


    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    training_loss = tf.summary.scalar('Training_Loss', loss_op)
    training_accuracy = tf.summary.scalar('Training_Acc', accuracy)

    # Initialize the variables (i.e. assign their default value)
    #init = tf.global_variables_initializer()
    global_step = tf.train.get_or_create_global_step()
    #learning_rate = tf.train.exponential_decay(learning_rate, global_step,1000, 0.96, staircase=True)

    train_step = (
            tf.train.AdamOptimizer(learning_rate)
            .minimize(loss_op, global_step=global_step))
    
    
   


# In[ ]:


# def run_train_epoch(target,FLAGS,epoch_index):
#     """Restores the last checkpoint and runs a training epoch
#     Inputs:
#         - target: device setter for distributed work
#         - FLAGS:
#             - requires FLAGS.logs_dir from which the model will be restored.
#             Note that whatever most recent checkpoint from that directory will be used.
#             - requires FLAGS.steps_per_epoch
#         - epoch_index: index of current epoch
#     """

#     hooks=[tf.train.StopAtStepHook(last_step=FLAGS.steps_per_epoch*epoch_index)] # Increment number of required training steps
#     i = 1

with tf.train.MonitoredTrainingSession(master=target,
is_chief=(FLAGS.task_index == 0),
checkpoint_dir=FLAGS.logs_dir
#    ,hooks = hooks
) as sess:
    

#         while not sess.should_stop():
#             variables = [loss_op, learning_rate, train_step]
#             current_loss, lr, _ = sess.run(variables)

#             print("Iteration %s - Batch loss: %s" % ((epoch_index)*FLAGS.steps_per_epoch + i,current_loss))
#             i+=1

# for e in range(FLAGS.nb_epochs):
#     run_train_epoch(target, FLAGS, e)
    for i in range(2000):
        batch_x, batch_y = next_batch(batch_size,X_train,y_trainHot)
        [train_accuracy, l] = sess.run([accuracy, loss_op], feed_dict={x: batch_x, y: batch_y})
        if FLAGS.task_index == 0:
            if i % 5 == 0:
                print("Batch %s - training accuracy: %s" % (i,train_accuracy))
                  #writer.add_summary(s, i)
#                 if i % 500 == 0:
#                     sess.run(assignment, feed_dict={x: mnist.test.images[:1024], y: mnist.test.labels[:1024]})
        sess.run(train_step, feed_dict={x: batch_x, y: batch_y})


# In[ ]:


# with tf.train.MonitoredTrainingSession(
#       master=target,
#       is_chief=(0),
#       checkpoint_dir="logs/") as sess:
#     while not sess.should_stop():
#           sess.run(train_step)

