#Classify the MNIST digits using a one nearest neighbour classifier and Euclidean distance

from scipy import sparse
import numpy as np
from array import array

import struct
import os
import time
import gzip

MNIST_prefix="http://yann.lecun.com/exdb/mnist/"


#This PULLMNIST function is from https://gist.github.com/akesling/5358964

def PullMNIST(dataset = "training", source_path = ".", save_dir = None):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    
    dataset (string): one of 'training' or 'testing'
    
    source_path (string): location of MNIST data set.  Can be a URL if you are 
    fetching the data for the first time, or a local path where the data resides
    
    save_dir (string): [optional]  the local path where you'd like to save the 
    MNIST data if fetching it from a URL
    """

    if dataset is "training":
        fname_img = os.path.join(source_path, 'train-images-idx3-ubyte.gz')
        fname_lbl = os.path.join(source_path, 'train-labels-idx1-ubyte.gz')
    elif dataset is "testing":
        fname_img = os.path.join(source_path, 't10k-images-idx3-ubyte.gz')
        fname_lbl = os.path.join(source_path, 't10k-labels-idx1-ubyte.gz')
    else:
        raise ValueError("dataset must be \'testing\' or \'training\'")

    # Take care of gzipped files transparently
    if fname_lbl.endswith('gz') and fname_img.endswith('gz'):
        myopen = gzip.open
    else:
        myopen = open    

    
    if fname_lbl.startswith("http") and fname_lbl.startswith("http"):
        
        # Fetch files over http, save, convert, return np arrays
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        import urllib.request
        labels, info = urllib.request.urlretrieve(fname_lbl, os.path.join(os.path.dirname(fname_lbl), save_dir, os.path.basename(fname_lbl)))      
        images, info = urllib.request.urlretrieve(fname_img, os.path.join(os.path.dirname(fname_img), save_dir, os.path.basename(fname_img)))
           
        with myopen(os.path.join(save_dir, os.path.basename(fname_lbl)), 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            my_lbl = array("B", flbl.read())
            lbl = np.array(my_lbl, dtype=np.int8)
    
        with myopen(os.path.join(save_dir, os.path.basename(fname_img)), 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            my_img = array("B", fimg.read())
            
            images = []
            for i in range(num):
                images.append([0] * rows * cols)
        
            for i in range(num):
                images[i][:] = my_img[i * rows * cols:(i + 1) * rows * cols]
                    
            img = np.array(images)       
        
    else:
        
        # Load from local filesystem, convert, return np arrays
        with myopen(fname_lbl, 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            my_lbl = array("B", flbl.read())
            lbl = np.array(my_lbl, dtype=np.int8)
    
        with myopen(fname_img, 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            my_img = array("B", fimg.read())
            
            images = []
            for i in range(num):
                images.append([0] * rows * cols)
        
            for i in range(num):
                images[i][:] = my_img[i * rows * cols:(i + 1) * rows * cols]
                    
            img = np.array(images)

    return {'labels':lbl,'images':img}
        
# Pull the MNIST data from yann.lecun.com

#Training = PullMNIST("training", MNIST_prefix, os.path.expanduser("~/projects/pyprobml/data/MNIST"))
#Testing = PullMNIST("testing", MNIST_prefix, os.path.expanduser("~/projects/pyprobml/data/MNIST"))

# Pull the MNIST data from a local dir
Training = PullMNIST("training", os.path.expanduser("~/projects/pyprobml/data/MNIST"))
TrainLabels = Training['labels']
TrainIms = Training['images'].astype('int32')

Testing = PullMNIST("testing", os.path.expanduser("~/projects/pyprobml/data/MNIST"))
TestLabels = Testing['labels']
TestIms = Testing['images'].astype('int32')

del Testing, Training

#Flattens images into sparse vectors. So we go from 3D to 2D image datasets.
def Flatten(Ims):
    return(sparse.csr_matrix(Ims.reshape(Ims.shape[0],-1)))

TrainIms = Flatten(TrainIms)
TestIms = Flatten(TestIms)

## SAMPLING - In case we want to apply this to a subset of the data
#TestS = 1000 #Size of test data
#TrainS = 10000  #Size of training data
#TestLabels = TestLabels[:TestS]
#TestIms = TestIms[:TestS,:]
#TrainLabels = TrainLabels[:TrainS]
#TrainIms = TrainIms[:TrainS,:]

t0 = time.time()
#Calculating squared vector norms
TrainNorms = np.array([TrainIms[i,:]*TrainIms[i,:].T.toarray() for i in range(TrainIms.shape[0])]).reshape(-1,1)

def PredictandError(testims,testlabs):
    #This is not technically a distance - we are leaving out the Test squared norms because they are constant 
    #when determining a nearest neighbor.
    Distances = TrainNorms*np.ones(testims.shape[0]).T - 2*TrainIms*testims.T

    predictions = TrainLabels[np.argmin(Distances,axis=0)]

    error = 1 - np.mean(np.equal(predictions,testlabs))
    return(error*100)

BucketSize = 1000
errors = []
for i in range(0,len(TestLabels),BucketSize):
    errors.append(PredictandError(TestIms[i:(i+BucketSize)],TestLabels[i:(i+BucketSize)]))

t1 = time.time()
print('error:' + str(np.mean(errors))) #Since the buckets are equal size, we can average their errors.
print('Time taken:' + str(t1-t0))
