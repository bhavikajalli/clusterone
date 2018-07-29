
# coding: utf-8

# In[9]:


from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import tensorflow as tf
from PIL import Image
import os
import tensorflow as tf
from clusterone import get_data_path, get_logs_path


# In[15]:


get_ipython().magic('matplotlib inline')
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144


# In[8]:


train_dir = "asl_alphabet_train/"
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
                img_file = Image.open(folder + folderName + '/' + image_filename)
                if img_file is not None:
                    #img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3))
                    img_arr = np.asarray(img_file)
                    X.append(img_arr)
                    y.append(label)
    X = np.asarray(X)
    y = np.asarray(y)
    return X,y
X_train, y_train = get_data(train_dir) 
#X_test, y_test= get_data(test_dir) # Too few images


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2) 

# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

y_trainHot = to_categorical(y_train, num_classes = 30)
y_testHot = to_categorical(y_test, num_classes = 30)


# In[10]:


###Dataset Creation
# Shuffle data to permit further subsampling
from sklearn.utils import shuffle
X_train, y_trainHot = shuffle(X_train, y_trainHot, random_state=42)
print("Shuffling data")
X_test, y_testHot = shuffle(X_test, y_testHot, random_state=42)
print("Data has been shuffled for further subsampling")
X_train = X_train[:30000]
X_test = X_test[:30000]
y_trainHot = y_trainHot[:30000]
y_testHot = y_testHot[:30000]


# In[18]:


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
plotHistogram(X_train[1])
X_train.shape


# In[20]:


##Defining the Model
#Training using a simple model
learning_rate = 0.001
num_steps = 2000
training_epochs = 2
batch_size = 128
display_step = 100

# Network Parameters
n_input = 200 # ASL data input (img shape: 200*200*3)
n_classes = 30 # ASL total classes (0-25 alphabets, 1 space,del,nothing and other digits)
dropout = 0.5


# In[30]:


#Defining and building the graph
def conv_net(x, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        #The ASL data input is a 3-D vector of 200x200 features
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 200, 200, 3])

        # Convolution Layer with 64 filters and a kernel size of 5
        x = tf.layers.conv2d(x, 64, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        x = tf.layers.max_pooling2d(x, 2, 2)

        # Convolution Layer with 256 filters and a kernel size of 5
        x = tf.layers.conv2d(x, 256, 3, activation=tf.nn.relu)
        # Convolution Layer with 512 filters and a kernel size of 5
        x = tf.layers.conv2d(x, 512, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        x = tf.layers.max_pooling2d(x, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        x = tf.contrib.layers.flatten(x)

        x = tf.layers.dense(x, 2048)
        # Apply Dropout (if is_training is False, dropout is not applied)
        x = tf.layers.dropout(x, rate=dropout, training=is_training)

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


# In[31]:


def next_batch(batch_size, data, labels):
    '''
    Return a total of `batch_size` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


# In[38]:


##Graph Creation and Training for the distributed training platform
try:
    job_name = os.environ['JOB_NAME']
    task_index = os.environ['TASK_INDEX']
    ps_hosts = os.environ['PS_HOSTS']
    worker_hosts = os.environ['WORKER_HOSTS']
except: # we are not on TensorPort, assuming local, single node
    task_index = 0
    ps_hosts = None
    worker_hosts = None
        
        
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

# Defining graph
# # tf Graph input
x = tf.placeholder("float", [None, n_input,n_input,3])
y = tf.placeholder("float", [None, n_classes])
reuse_vars = False

with tf.device(device):
        #TODO define your graph here
#     logits = conv_net(x,n_classes, dropout,reuse=reuse_vars ,is_training=True)

#     # Define loss and optimizer
#     loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
#     logits=logits, labels=y))
#     optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#     train_op = optimizer.minimize(loss_op)

#     # Evaluate model (with test logits, for dropout to be disabled)
#     correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#     # Initialize the variables (i.e. assign their default value)
#     init = tf.global_variables_initializer()
    logits_train = conv_net(x, n_classes, dropout,
                            reuse=reuse_vars, is_training=True)
    # Create another graph for testing that reuse the same weights
    logits_test = conv_net(x, n_classes, dropout,
                           reuse=True, is_training=False)

    # Define loss and optimizer (with train logits, for dropout to take effect)
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits_train, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    grads = optimizer.compute_gradients(loss_op)

    # Only first GPU compute accuracy
    if i == 0:
        # Evaluate model (with test logits, for dropout to be disabled)
        correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    reuse_vars = True
        

#Defining the number of training steps
hooks=[tf.train.StopAtStepHook(last_step=100)]

with tf.train.MonitoredTrainingSession(master=target,
    is_chief=(FLAGS.task_index == 0),
    checkpoint_dir=FLAGS.logs_dir,
    hooks = hooks) as sess:

    while not sess.should_stop():
            # execute training step here (read data, feed_dict, session)
            # TODO define training ops
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(len(X_train)/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = next_batch(batch_size,X_train,y_trainHot)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, loss_op], feed_dict={x: batch_x,
                                                              y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=",                     "{:.9f}".format(avg_cost))
        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: X_test, y: y_testHot}))


# In[37]:


logits = conv_net(x,n_classes, dropout,reuse=True ,is_training=True)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(len(X_train)/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = next_batch(batch_size,X_train,y_trainHot)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, loss_op], feed_dict={x: batch_x,
                                                              y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=",                     "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: X_test, y: y_testHot}))

