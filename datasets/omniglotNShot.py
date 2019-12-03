##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Albert Berenguel
## Computer Vision Center (CVC). Universitat Autonoma de Barcelona
## Email: aberenguel@cvc.uab.es
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from datasets import omniglot
import torchvision.transforms as transforms
from PIL import Image
import os.path
import json
import math

from numpy import array
import numpy as np
#moved inside constructor
#np.random.seed(2191)  # for reproducibility

# LAMBDA FUNCTIONS
filenameToPILImage = lambda x: Image.open(x).convert('L')
PiLImageResize = lambda x: x.resize((28,28))
np_reshape = lambda x: np.reshape(x, (28, 28, 1))

class OmniglotNShotDataset():
    def __init__(self, dataroot, batch_size = 100, classes_per_set=10, samples_per_class=1, is_use_sample_data = True, input_file="", input_labels_file="", total_input_files=-1, is_evaluation_only = False, evaluation_input_file = "", evaluation_labels_file = "", evaluate_classes = 1, is_eval_with_train_data = 0, negative_test_offset = 0, is_apply_pca_first = 0, cache_samples_for_evaluation = 100):

        if is_evaluation_only == False:
            np.random.seed(2191)  # for reproducibility
        else:
            #for variational testing 
            np.random.seed( np.random.randint(0, 1000)  )  
            
        if is_use_sample_data:
            if not os.path.isfile(os.path.join(dataroot,'data.npy')):
                self.x = omniglot.OMNIGLOT(dataroot, download=True,
                                         transform=transforms.Compose([filenameToPILImage,
                                                                       PiLImageResize,
                                                                       np_reshape]))

                """
                # Convert to the format of AntreasAntoniou. Format [nClasses,nCharacters,28,28,1]
                """
                temp = dict()
                for (img, label) in self.x:
                    if label in temp:
                        temp[label].append(img)
                    else:
                        temp[label]=[img]
                self.x = [] # Free memory

                for classes in temp.keys():
                    self.x.append(np.array(temp[ list(temp.keys())[classes]]))
                self.x = np.array(self.x)
                temp = [] # Free memory
                np.save(os.path.join(dataroot,'data.npy'),self.x)
            else:
                self.x = np.load(os.path.join(dataroot,'data.npy'))
        else:   
            self.x = [] 
            self.x_to_be_predicted = [] 
            self.x_to_be_predicted_cls_indexes = {} 
            
            self.prediction_classes = 9
            self.total_base_classes = 56
            
            base_classes_file = input_file+"_base_classes.json"
            self.evaluate_classes = evaluate_classes
            self.is_eval_with_train_data = True if is_eval_with_train_data == 1 else False
            self.negative_test_offset = negative_test_offset
                        
            #
            if is_evaluation_only == False or not os.path.exists( base_classes_file ):
                print( "(!) Merging inputs, should only be executed in training mode." )
                input = []
                input_labels = []
                print("total_input_files")
                print(total_input_files)
                for i in range(0, total_input_files):
                    print("total_input_files i " + str(i))
                    if i == 0:
                        input = array( json.load( open( input_file.replace('{i}', str(i)) ) ) ) 
                        input_labels = array( json.load( open( input_labels_file.replace('{i}', str(i)) ) ) ) 
                    else:
                        input = np.concatenate( ( input, array( json.load( open( input_file.replace('{i}', str(i)) ) ) ) ), axis=0 )
                        input_labels = np.concatenate( ( input_labels, array( json.load( open( input_labels_file.replace('{i}', str(i)) ) ) ) ), axis=0 )
                
                temp = dict()
                temp_to_be_predicted = dict()
                sizei = len(input)
                print("sizei")
                print(sizei)
                for i in np.arange(sizei):
                    #if is_evaluation_only == True and input_labels[i] >= self.prediction_classes:
                    #    continue

                    if input_labels[i] >= self.total_base_classes:
                        continue
                    
                    if input_labels[i] in temp:
                        if len( temp[input_labels[i]] ) >= 19:  #only 20 samples per class
                            if is_evaluation_only == False and (input_labels[i] < self.total_base_classes or np.mod( input_labels[i] - self.total_base_classes, 30 ) == 0 or np.mod( input_labels[i] - (self.total_base_classes+1), 30 ) == 0):            #True or False and (True or input_labels[i] == 6):
                                lbl_val = input_labels[i]
                                if input_labels[i] >= self.total_base_classes and np.mod( input_labels[i] - self.total_base_classes, 30 ) == 0:
                                    lbl_val = self.total_base_classes + int( (input_labels[i] - self.total_base_classes) / 30 )
                                if input_labels[i] >= self.total_base_classes and np.mod( input_labels[i] - (self.total_base_classes+1), 30 ) == 0:
                                    lbl_val = (self.total_base_classes*2) + int( (input_labels[i] - (self.total_base_classes+1)) / 30 )								
                                    
                                if lbl_val in temp_to_be_predicted:
                                    if len( temp_to_be_predicted[lbl_val] ) >= 10:  #only 20 samples per class
                                        continue
                                    
                                    temp_to_be_predicted[lbl_val].append( input[i][:,:,np.newaxis] )
                                else:     
                                    temp_to_be_predicted[lbl_val]=[input[i][:,:,np.newaxis]]
                        
                            continue
                        
                        temp[input_labels[i]].append( input[i][:,:,np.newaxis] )
                    else:
                        temp[input_labels[i]]=[input[i][:,:,np.newaxis]]

                #print( "temp.keys()" )
                unique, counts = np.unique(input_labels, return_counts=True)
                #print( dict(zip(unique, counts)) )
                
                input = []  # Free memory
                input_labels = []  # Free memory
                self.x = [] # Free memory

                        
                for classes in temp.keys():
                    self.x.append(np.array(temp[ list(temp.keys())[classes]]))
                self.x = np.array(self.x)
                temp = [] # Free memory

                #np.save(os.path.join(dataroot,'data.npy'),self.x)
                with open( base_classes_file, 'w') as outfile:
                    json.dump(self.x.tolist(), outfile)                      
            else:
                print("loaded prepared base_classes_file")
                self.x = array( json.load( open( base_classes_file ) ) ) 
                
                
            if is_evaluation_only == False:
                print( "temp_to_be_predicted.keys()" )
                print( temp_to_be_predicted.keys() )
                cls_index = 0
                for classes in temp_to_be_predicted.keys():
                    self.x_to_be_predicted_cls_indexes[classes] = cls_index
                    self.x_to_be_predicted.append(np.array(temp_to_be_predicted[ list(temp_to_be_predicted.keys())[classes]]))
                    cls_index = cls_index + 1
                self.x_to_be_predicted = np.array(self.x_to_be_predicted)
                temp_to_be_predicted = [] # Free memory
                
                #np.save(os.path.join(dataroot,'data.npy'),self.x)
                with open( base_classes_file+"_x_to_be_predicted.json", 'w') as outfile:
                    json.dump(self.x_to_be_predicted.tolist(), outfile)                      
                    

            #
            if is_evaluation_only == True:
            
                if not os.path.exists(evaluation_input_file.replace('{i}', str(0)) + "_prepared.json"):
                    input = array( json.load( open( evaluation_input_file.replace('{i}', str(0)) ) ) ) 
                    input_labels = array( json.load( open( evaluation_labels_file.replace('{i}', str(0)) ) ) ) 
                    
                    temp = dict()
                    temp_to_be_predicted = dict()
                    sizei = len(input)
                    print("sizei")
                    print(sizei)
                    for i in np.arange(sizei):
                        if input_labels[i] in temp:
                            if len( temp[input_labels[i]] ) >= 19:  #only 20 samples per class
                                if is_evaluation_only == False and (input_labels[i] < self.total_base_classes or np.mod( input_labels[i] - self.total_base_classes, 30 ) == 0 or np.mod( input_labels[i] - (self.total_base_classes+1), 30 ) == 0):            #True or False and (True or input_labels[i] == 6):
                                    lbl_val = input_labels[i]
                                    if input_labels[i] >= self.total_base_classes and np.mod( input_labels[i] - self.total_base_classes, 30 ) == 0:
                                        lbl_val = self.total_base_classes + int( (input_labels[i] - self.total_base_classes) / 30 )
                                    if input_labels[i] >= self.total_base_classes and np.mod( input_labels[i] - (self.total_base_classes+1), 30 ) == 0:
                                        lbl_val = (self.total_base_classes*2) + int( (input_labels[i] - (self.total_base_classes+1)) / 30 )								
                                        
                                    if lbl_val in temp_to_be_predicted:
                                        if len( temp_to_be_predicted[lbl_val] ) >= 10:  #only 20 samples per class
                                            continue
                                        
                                        temp_to_be_predicted[lbl_val].append( input[i][:,:,np.newaxis] )
                                    else:     
                                        temp_to_be_predicted[lbl_val]=[input[i][:,:,np.newaxis]]
                            
                                continue
                            
                            temp[input_labels[i]].append( input[i][:,:,np.newaxis] )
                        else:
                            temp[input_labels[i]]=[input[i][:,:,np.newaxis]]

                    print( "temp.keys()" )
                    #print( temp.keys() )
                    #for key, value in temp.items(): 
                    #    if True or len(value) < 19:
                    #        print("key " + str(key) + " len " + str(len(value)))
                    unique, counts = np.unique(input_labels, return_counts=True)
                    print( dict(zip(unique, counts)) )
                    
                    input = []  # Free memory
                    input_labels = []  # Free memory
                    
                    self.evaluation = [] 
                    print(temp.keys())
                    for classes in temp.keys():
                        self.evaluation.append(np.array(temp[ list(temp.keys())[classes]]))
                    self.evaluation = np.array(self.evaluation)
                    temp = [] # Free memory
                else:
                    print("loaded prepared evaluation_input_file")
                    self.evaluation = array( json.load( open( evaluation_input_file.replace('{i}', str(0)) + "_prepared.json" ) ) ) 
                    
            
        
        #TODO tmp. compare 
        """
        print(self.x.shape)
        print(self.evaluation.shape)
        is_found = False
        for h in range(0, self.x.shape[0]):
            print("classssssssssssssssssssssssssssssssssssssssssssssssssssssssssss " + str(h))
            for i in range(0, self.x.shape[1]):
                #print( "x indices val " + str(self.x[h,i,27,99,0]) + " " + str(self.x[h,i,27,103,0]) + " " + str(self.x[h,i,27,107,0]) )
                xt = np.copy(self.x[h,i,:,:,:])
                xt[27,99,0] = 0
                xt[27,103,0] = 0
                xt[27,107,0] = 0
                et = np.copy(self.evaluation[self.evaluate_classes,0,:,:,:])
                et[27,99,0] = 0
                et[27,103,0] = 0
                et[27,107,0] = 0
                #print( "evaluation indices val " + str(self.evaluation[self.evaluate_classes,i,27,99,0]) + " " + str(self.evaluation[self.evaluate_classes,i,27,103,0]) + " " + str(self.evaluation[self.evaluate_classes,i,27,107,0]) )
                result = np.subtract( xt, et)
                if (result > 1.0).sum() >= 1 or (result < -1.0).sum() >= 1:
                    continue
                print ('the difference h ' + str(h) + ' i ' + str(i))
                print (result)
                if (result > 0.0).sum() == 0 and (result < 0.0).sum() == 0:
                    is_found = True
                    self.evaluate_classes = h
                    self.evaluation[self.evaluate_classes,:,27,99,0] = self.x[h,i,27,99,0]
                    self.evaluation[self.evaluate_classes,:,27,103,0] = self.x[h,i,27,103,0]
                    self.evaluation[self.evaluate_classes,:,27,107,0] = self.x[h,i,27,107,0]
                    print("fioundddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd")
                    break
                    
            if is_found == True:
                break
            
        if is_found == False:
            sdfhsdhfkjhd
        
        if is_evaluation_only == True:
            is_found = False
            for i in range(0, self.x.shape[1]):
                xt = np.copy(self.x[self.evaluate_classes,i,:,:,:])
                xt[27,99,0] = 0
                xt[27,103,0] = 0
                xt[27,107,0] = 0
                et = np.copy(self.evaluation[self.evaluate_classes,0,:,:,:])
                et[27,99,0] = 0
                et[27,103,0] = 0
                et[27,107,0] = 0

                result = np.subtract( xt, et)
                if (result > 1.0).sum() >= 1 or (result < -1.0).sum() >= 1:
                    continue
                
                #print ('the difference i ' + str(i))
                #print (result)
                
                if (result > 0.0).sum() == 0 and (result < 0.0).sum() == 0:
                    is_found = True
                    self.evaluation[:,:,27,99,0] = self.x[self.evaluate_classes,i,27,99,0]
                    self.evaluation[:,:,27,103,0] = self.x[self.evaluate_classes,i,27,103,0]
                    self.evaluation[:,:,27,107,0] = self.x[self.evaluate_classes,i,27,107,0]
                    print("fioundddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd")
                    break
                
            if is_found == False:
                sdfhsdhfkjhd
        """
        
        #TODO temp
        self.x[:,:,27,99,0] = 0
        self.x[:,:,27,103,0] = 0
        self.x[:,:,27,107,0] = 0
        self.evaluation[:,:,27,99,0] = 0
        self.evaluation[:,:,27,103,0] = 0
        self.evaluation[:,:,27,107,0] = 0

        #
        self.shuffle_classes = np.arange(self.x.shape[0])
        self.is_apply_pca_first = is_apply_pca_first
        
        #pca 
        if self.is_apply_pca_first == 1:
            data = self.x.reshape(self.x.shape[0]*self.x.shape[1], self.x.shape[2]*self.x.shape[3])
        
            ##
            #print("pca matlab")
            #from matplotlib.mlab import PCA
            #p = PCA(data)
            #print( p.Wt )
            #print( p.Wt.shape )
            

            #
            print( "pca custom from so https://stackoverflow.com/a/13224592" )
            def PCA(data, dims_rescaled_data=2):
                """
                returns: data transformed in 2 dims/columns + regenerated original data
                pass in: data as 2D NumPy array
                """
                import numpy as NP
                from scipy import linalg as LA
                m, n = data.shape
                # mean center the data
                data -= data.mean(axis=0)
                # calculate the covariance matrix
                R = NP.cov(data, rowvar=False)
                # calculate eigenvectors & eigenvalues of the covariance matrix
                # use 'eigh' rather than 'eig' since R is symmetric, 
                # the performance gain is substantial
                evals, evecs = LA.eigh(R)
                # sort eigenvalue in decreasing order
                idx = NP.argsort(evals)[::-1]
                evecs = evecs[:,idx]
                # sort eigenvectors according to same index
                evals = evals[idx]
                # select the first n eigenvectors (n is desired dimension
                # of rescaled data array, or dims_rescaled_data)
                evecs = evecs[:, :dims_rescaled_data]
                # carry out the transformation on the data using eigenvectors
                # and return the re-scaled data, eigenvalues, and eigenvectors
                return NP.dot(evecs.T, data.T).T, evals, evecs

            def test_PCA(data, dims_rescaled_data=2):
                '''
                test by attempting to recover original data array from
                the eigenvectors of its covariance matrix & comparing that
                'recovered' array with the original data
                '''
                _ , _ , eigenvectors = PCA(data, dim_rescaled_data=2)
                data_recovered = NP.dot(eigenvectors, m).T
                data_recovered += data_recovered.mean(axis=0)
                assert NP.allclose(data, data_recovered)

            def plot_pca(data):
                from matplotlib import pyplot as MPL
                clr1 =  '#2026B2'
                fig = MPL.figure()
                ax1 = fig.add_subplot(111)
                data_resc, data_orig = PCA(data)
                ax1.plot(data_resc[:, 0], data_resc[:, 1], '.', mfc=clr1, mec=clr1)
                MPL.show()

            #print( plot_pca(data) )
            print(data.shape)
            
            
            from sklearn.decomposition import PCA
            p = PCA(n_components = 728).fit_transform(data)
            print( type(p) )
            print( p )
            print( p.shape )
            
            
        """
        #TODO temp
        self.x = self.x[:30]
        self.evaluation = self.evaluation[0:30]
        shuffle_classes = np.arange(self.x.shape[0])
        np.random.shuffle(shuffle_classes)
        print("shuffle_classes")
        print(shuffle_classes)
        self.shuffle_classes = shuffle_classes
        self.x = self.x[shuffle_classes]
        self.evaluation = self.evaluation[shuffle_classes]
        """
                        
        self.data_pack_shape_2 = None
        self.data_pack_shape_3 = None
        
        
        """
        Constructs an N-Shot omniglot Dataset
        :param batch_size: Experiment batch_size
        :param classes_per_set: Integer indicating the number of classes per set
        :param samples_per_class: Integer indicating samples per class
        e.g. For a 20-way, 1-shot learning task, use classes_per_set=20 and samples_per_class=1
             For a 5-way, 10-shot learning task, use classes_per_set=5 and samples_per_class=10
        """

        #shuffle_classes = np.arange(self.x.shape[0])
        #np.random.shuffle(shuffle_classes)
        #self.x = self.x[shuffle_classes]
        self.cache_sample = 0
        self.cache_sample_prediction = 0
        self.is_rotate = False
        if is_use_sample_data:
            self.is_rotate = True
            self.cache_sample = 1000
            self.cache_sample_prediction = 10
            self.x_train, self.x_test, self.x_val  = self.x[:1200], self.x[1200:1500], self.x[1500:]
        else:
            self.is_rotate = False
            self.cache_sample = 300
            self.cache_sample_prediction = cache_samples_for_evaluation  
            if is_evaluation_only == False:
                #self.x_train, self.x_test, self.x_val  = self.x[:900], self.x[900:1200], self.x[1200:]
                self.x_train, self.x_test, self.x_val  = self.x[:30], self.x[30:43], self.x[43:]
            else:
                self.x_train  = self.x[:]
         
        #print( self.x_train[0][0] )
        self.normalization()
        #print( self.x_train[0][0] )
 
        self.batch_size = batch_size
        self.n_classes = self.x.shape[0]
        self.classes_per_set = classes_per_set
        self.samples_per_class = samples_per_class

        if is_evaluation_only == False:
            self.indexes = {"train": 0, "val": 0, "test": 0, "x_to_be_predicted": 0}
            self.datasets = {"train": self.x_train, "val": self.x_val, "test": self.x_test, "x_to_be_predicted": self.x_to_be_predicted} #original data cached
            self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"], ""),  #current epoch data cached
                                   "val": self.load_data_cache(self.datasets["val"], ""),
                                   "test": self.load_data_cache(self.datasets["test"], ""),
                                   "x_to_be_predicted": self.load_data_cache(self.datasets["x_to_be_predicted"], "x_to_be_predicted")}
        else:
            self.indexes = {"evaluation": 0}
            self.datasets = {"evaluation": self.x_train} #original data cached
            self.datasets_cache = {"evaluation": self.load_data_cache_for_evaluation(self.datasets["evaluation"], "evaluation", self.evaluation)}
                                   
    def normalization(self):
        """
        Normalizes our data, to have a mean of 0 and sdt of 1
        """
        return 
        
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)
        print("train_shape", self.x_train.shape, "test_shape", self.x_test.shape, "val_shape", self.x_val.shape, "x_to_be_predicted", self.x_to_be_predicted.shape)
        print("before_normalization", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)
        
        #if required for your data enable normatlization by uncommenting below code 
        """
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_val = (self.x_val - self.mean) / self.std
        self.x_test = (self.x_test - self.mean) / self.std
        self.x_to_be_predicted = (self.x_to_be_predicted - self.mean) / self.std
        
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)
        """
        
        print("after_normalization", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)

    def load_data_cache(self, data_pack, data_pack_type):
        """
        Collects 1000 batches data for N-shot learning
        :param data_pack: Data pack to use (any one of train, val, test)
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        print( "data_pack" )
        print( data_pack_type )        
        print( data_pack.shape )
        
        """
        print( data_pack.shape[0] )
        print( data_pack.shape[2] )
        print( data_pack.shape[3] )
        """
        if self.data_pack_shape_2 == None:
            self.data_pack_shape_2 = data_pack.shape[2]
        if self.data_pack_shape_3 == None:
            self.data_pack_shape_3 = data_pack.shape[3]            
        
        n_samples = self.samples_per_class * self.classes_per_set
        data_cache = []
        for sample in range(self.cache_sample):
        
            """
            #TODO temp. profiling, comment it when not needed
            import cProfile, pstats
            import io as StringIO
            print( "profiling start" )
            pr = cProfile.Profile()
            pr.enable()
            """
        
            support_set_x = np.zeros((self.batch_size, n_samples, self.data_pack_shape_2, self.data_pack_shape_3, 1))
            support_set_y = np.zeros((self.batch_size, n_samples))
            target_x = np.zeros((self.batch_size, self.samples_per_class, self.data_pack_shape_2, self.data_pack_shape_3, 1), dtype=np.int)
            target_y = np.zeros((self.batch_size, self.samples_per_class), dtype=np.int)
            for i in range(self.batch_size):
                pinds = np.random.permutation(n_samples)
                classes = np.random.choice(data_pack.shape[0], self.classes_per_set, False if not data_pack_type == "x_to_be_predicted" else False)  #False
                # select 1-shot or 5-shot classes for test with repetition
                x_hat_class = np.random.choice(classes, self.samples_per_class, True)
                pinds_test = np.random.permutation(self.samples_per_class)
                ind = 0
                ind_test = 0
                for j, cur_class in enumerate(classes):  # each class
                    #print( "example_inds" )
                    if cur_class in x_hat_class:
                        # Count number of times this class is inside the meta-test
                        n_test_samples = np.sum(cur_class == x_hat_class)
                        example_inds = np.random.choice(data_pack.shape[1], self.samples_per_class + n_test_samples, False)
                        #print( "example_inds here 1 " + str(n_test_samples) )
                    else:
                        #print( "example_inds here 2 " )
                        example_inds = np.random.choice(data_pack.shape[1], self.samples_per_class, False)

                    #print( example_inds )
                        
                    # meta-training
                    for eind in example_inds[:self.samples_per_class]:
                        support_set_x[i, pinds[ind], :, :, :] = data_pack[cur_class][eind]
                        support_set_y[i, pinds[ind]] = j
                        ind = ind + 1
                    # meta-test
                    for eind in example_inds[self.samples_per_class:]:
                        """
                        print( "eind" )
                        print( eind )
                        print( cur_class )
                        print( i )
                        print( ind_test )
                        print( pinds_test[ind_test] )
                        """
                        
                        target_x[i, pinds_test[ind_test], :, :, :] = data_pack[cur_class][eind]
                        target_y[i, pinds_test[ind_test]] = j
                        ind_test = ind_test + 1

            data_cache.append([support_set_x, support_set_y, target_x, target_y])
            
            """
            #TODO temp. profiling, comment it when not needed
            pr.disable()
            s = StringIO.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print( s.getvalue() )
            sdfkjhskdfhkshdf
            """
            
        return data_cache

    def load_data_cache_for_evaluation(self, data_pack, data_pack_type, data_pack_evaluation):
        """
        Collects 1000 batches data for N-shot learning
        :param data_pack: Data pack to use (any one of train, val, test)
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        print( "data_pack" )
        print( data_pack_type )        
        print( data_pack.shape )
        
        """
        print( data_pack.shape[0] )
        print( data_pack.shape[2] )
        print( data_pack.shape[3] )
        """
        if self.data_pack_shape_2 == None:
            self.data_pack_shape_2 = data_pack.shape[2]
        if self.data_pack_shape_3 == None:
            self.data_pack_shape_3 = data_pack.shape[3]            
        
        #TODO temp. eval with train data
        is_eval_with_train_data = self.is_eval_with_train_data
        
        n_samples = self.samples_per_class * self.classes_per_set
        data_cache = []
        for sample in range(0, self.cache_sample_prediction):
        
            """
            #TODO temp. profiling, comment it when not needed
            import cProfile, pstats
            import io as StringIO
            print( "profiling start" )
            pr = cProfile.Profile()
            pr.enable()
            """
            
            self.evaluate_classes = math.floor(sample / 10)
        
            support_set_x = np.zeros((self.batch_size, n_samples, self.data_pack_shape_2, self.data_pack_shape_3, 1))
            support_set_y = np.zeros((self.batch_size, n_samples), dtype=np.int)#)
            target_x = np.zeros((self.batch_size, self.samples_per_class, self.data_pack_shape_2, self.data_pack_shape_3, 1))#, dtype=np.int)
            target_y = np.zeros((self.batch_size, self.samples_per_class), dtype=np.int)
            target_y_actuals = np.zeros((self.batch_size, self.samples_per_class), dtype=np.int)
            for i in range(self.batch_size):
                pinds = np.random.permutation(n_samples)
                #classes = np.random.choice(data_pack.shape[0], self.classes_per_set, False if not data_pack_type == "x_to_be_predicted" else False)  #False
                #classes = np.random.choice( self.prediction_classes, self.classes_per_set, False if not data_pack_type == "x_to_be_predicted" else False)  
                classes = np.random.choice( 30, self.classes_per_set, False if not data_pack_type == "x_to_be_predicted" else False)  
                
                # select 1-shot or 5-shot classes for test with repetition
                x_hat_class = np.random.choice(classes, self.samples_per_class, True)
                pinds_test = np.random.permutation(self.samples_per_class)
                ind = 0
                ind_test = 0
                for j, cur_class in enumerate(classes):  # each class
                    example_inds_test = []
                    #print( "example_inds j " + str(j) )
                    if cur_class in x_hat_class:
                        # Count number of times this class is inside the meta-test
                        n_test_samples = np.sum(cur_class == x_hat_class)
                        if is_eval_with_train_data == True or not cur_class == self.evaluate_classes:
                            if not cur_class == self.evaluate_classes:
                                example_inds = np.random.choice(data_pack.shape[1], self.samples_per_class+n_test_samples, False)
                            else:
                                #print( "example_inds_test here 1 in train mode" )
                                example_inds = np.random.choice(data_pack.shape[1], self.samples_per_class + (n_test_samples - 1), False)
                                example_inds_test = np.array( [0] ) #np.random.choice(self.evaluate_classes, self.evaluate_classes, False)
                        else:
                            #print( "example_inds_test here 1 " )
                            example_inds = np.random.choice(data_pack.shape[1], self.samples_per_class + (n_test_samples - 1), False)
                            example_inds_test = np.array( [0] ) #np.random.choice(self.evaluate_classes, self.evaluate_classes, False)
                            #print( "example_inds here 1 " + str(n_test_samples) )
                    else:
                        #print( "example_inds here 2 " )
                        example_inds = np.random.choice(data_pack.shape[1], self.samples_per_class, False)

                    #print( example_inds )
                         
                    # meta-training
                    for eind in example_inds[:self.samples_per_class]:
                        support_set_x[i, pinds[ind], :, :, :] = data_pack[cur_class][eind]
                        support_set_y[i, pinds[ind]] = j
                        ind = ind + 1
                    # meta-test
                    if is_eval_with_train_data == True and not cur_class == self.evaluate_classes:
                        for eind in example_inds[self.samples_per_class:]:
                            """
                            print( "eind" )
                            print( eind )
                            print( cur_class )
                            print( i )
                            print( ind_test )
                            print( pinds_test[ind_test] )
                            """
                            target_x[i, pinds_test[ind_test], :, :, :] = data_pack[cur_class][eind]
                            target_y[i, pinds_test[ind_test]] = j
                            ind_test = ind_test + 1
                    else:
                        for eind in example_inds[self.samples_per_class:]:
                            """
                            print( "eind" )
                            print( eind )
                            print( cur_class )
                            print( i )
                            print( ind_test )
                            print( pinds_test[ind_test] )
                            """
                            target_x[i, pinds_test[ind_test], :, :, :] = data_pack[cur_class][eind]
                            target_y[i, pinds_test[ind_test]] = j
                            ind_test = ind_test + 1
                    
                        if len(example_inds_test) > 0:
                            for eind in example_inds_test[:]:
                                """
                                print( "eind" )
                                print( eind )
                                print( cur_class )
                                print( i )
                                print( ind_test )
                                print( pinds_test[ind_test] )
                                """
                                
                                if is_eval_with_train_data == True:
                                    target_x[i, pinds_test[ind_test], :, :, :] = data_pack[cur_class+self.negative_test_offset][eind]
                                else:
                                    target_x[i, pinds_test[ind_test], :, :, :] = data_pack_evaluation[cur_class+self.negative_test_offset][eind]
                                target_y[i, pinds_test[ind_test]] = j
                                target_y_actuals[i, pinds_test[ind_test]] = (cur_class+1+self.negative_test_offset) * -1
                                ind_test = ind_test + 1

            data_cache.append([support_set_x, support_set_y, target_x, target_y, target_y_actuals])
            
            """
            #TODO temp. profiling, comment it when not needed
            pr.disable()
            s = StringIO.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print( s.getvalue() )
            sdfkjhskdfhkshdf
            """
            
        return data_cache
        
    def __get_batch(self, dataset_name):
        """
        Gets next batch from the dataset with name.
        :param dataset_name: The name of the dataset (one of "train", "val", "test")
        :return:
        """
        if self.indexes[dataset_name] >= len(self.datasets_cache[dataset_name]):
            self.indexes[dataset_name] = 0
            self.datasets_cache[dataset_name] = self.load_data_cache(self.datasets[dataset_name], dataset_name)
        next_batch = self.datasets_cache[dataset_name][self.indexes[dataset_name]]
        self.indexes[dataset_name] += 1
        x_support_set, y_support_set, x_target, y_target = next_batch
        return x_support_set, y_support_set, x_target, y_target

    def get_batch(self,str_type, rotate_flag = False):

        """
        Get next batch
        :return: Next batch
        """
        x_support_set, y_support_set, x_target, y_target = self.__get_batch(str_type)
        if rotate_flag:
            k = int(np.random.uniform(low=0, high=4))
            # Iterate over the sequence. Extract batches.
            for i in np.arange(x_support_set.shape[0]):
                x_support_set[i,:,:,:,:] = self.__rotate_batch(x_support_set[i,:,:,:,:],k)
            # Rotate all the batch of the target images
            for i in np.arange(x_target.shape[0]):
                x_target[i,:,:,:,:] = self.__rotate_batch(x_target[i,:,:,:,:], k)
        return x_support_set, y_support_set, x_target, y_target

    def get_batch_custom(self,str_type, cls, rotate_flag = False):
        """
        Get next batch
        :return: Next batch
        """
        x_support_set, y_support_set, x_target, y_target = self.__get_batch(str_type)
        if rotate_flag:
            k = int(np.random.uniform(low=0, high=4))
            # Iterate over the sequence. Extract batches.
            for i in np.arange(x_support_set.shape[0]):
                x_support_set[i,:,:,:,:] = self.__rotate_batch(x_support_set[i,:,:,:,:],k)
            # Rotate all the batch of the target images
            for i in np.arange(x_target.shape[0]):
                x_target[i,:,:,:,:] = self.__rotate_batch(x_target[i,:,:,:,:], k)

        """
        print( "get_batch_custom" )
        print( x_support_set.shape )
        print( y_support_set.shape )
        print( x_target.shape )
        print( y_target.shape )

        x_support_set_tmp, y_support_set_tmp, x_target_tmp, y_target_tmp = x_support_set, y_support_set, x_target, y_target

        for i in np.arange(8):
            x_support_set_tmp[i,:,:,:,:], y_support_set_tmp[i,:], x_target_tmp[i,:,:,:,:], y_target_tmp[i,:] = x_support_set[self.x_to_be_predicted_cls_indexes[cls]:self.x_to_be_predicted_cls_indexes[cls]+1,:,:,:,:], y_support_set[self.x_to_be_predicted_cls_indexes[cls]:self.x_to_be_predicted_cls_indexes[cls]+1,:], x_target[self.x_to_be_predicted_cls_indexes[cls]:self.x_to_be_predicted_cls_indexes[cls]+1,:,:,:,:], y_target[self.x_to_be_predicted_cls_indexes[cls]:self.x_to_be_predicted_cls_indexes[cls]+1,:]
        print( x_support_set_tmp.shape )
        print( y_support_set_tmp )
        print( x_target_tmp.shape )
        print( y_target_tmp )   
                
        return x_support_set_tmp, y_support_set_tmp, x_target_tmp, y_target_tmp
        """
        
        for i in np.arange( len(y_support_set) ):
            for j in np.arange( len(y_support_set[i]) ):
                if y_support_set[i][j] >= self.total_base_classes and y_support_set[i][j] < (self.total_base_classes*2):
                    y_support_set[i][j] = self.total_base_classes + ( (y_support_set[i][j] - self.total_base_classes) * 30 )
                if y_support_set[i][j] >= (self.total_base_classes*2):
                    y_support_set[i][j] = (self.total_base_classes+1) + ( (y_support_set[i][j] - (self.total_base_classes+1)) * 30 )
                
        for i in np.arange( len(y_target) ):
            for j in np.arange( len(y_target[i]) ):
                if y_target[i][j] >= self.total_base_classes and y_target[i][j] < (self.total_base_classes*2):
                    y_target[i][j] = self.total_base_classes + ( (y_target[i][j] - self.total_base_classes) * 30 )
                if y_target[i][j] >= (self.total_base_classes*2):
                    y_target[i][j] = (self.total_base_classes+1) + ( (y_target[i][j] - (self.total_base_classes+1)) * 30 )
					
        return x_support_set, y_support_set, x_target, y_target

    def __get_batch_evaluation(self, dataset_name):
        """
        Gets next batch from the dataset with name.
        :param dataset_name: The name of the dataset (one of "train", "val", "test")
        :return:
        """
        if self.indexes[dataset_name] >= len(self.datasets_cache[dataset_name]):
            self.indexes[dataset_name] = 0
            self.datasets_cache[dataset_name] = self.load_data_cache(self.datasets[dataset_name], dataset_name)
        next_batch = self.datasets_cache[dataset_name][self.indexes[dataset_name]]
        self.indexes[dataset_name] += 1
        x_support_set, y_support_set, x_target, y_target, target_y_actuals = next_batch
        return x_support_set, y_support_set, x_target, y_target, target_y_actuals
        
    def get_batch_evaluation(self,str_type, cls, rotate_flag = False):
        """
        Get next batch
        :return: Next batch
        """
        x_support_set, y_support_set, x_target, y_target, target_y_actuals = self.__get_batch_evaluation(str_type)
        if rotate_flag:
            k = int(np.random.uniform(low=0, high=4))
            # Iterate over the sequence. Extract batches.
            for i in np.arange(x_support_set.shape[0]):
                x_support_set[i,:,:,:,:] = self.__rotate_batch(x_support_set[i,:,:,:,:],k)
            # Rotate all the batch of the target images
            for i in np.arange(x_target.shape[0]):
                x_target[i,:,:,:,:] = self.__rotate_batch(x_target[i,:,:,:,:], k)
        """
        for i in np.arange( len(y_support_set) ):
            for j in np.arange( len(y_support_set[i]) ):
                if y_support_set[i][j] >= self.total_base_classes and y_support_set[i][j] < (self.total_base_classes*2):
                    y_support_set[i][j] = self.total_base_classes + ( (y_support_set[i][j] - self.total_base_classes) * 30 )
                if y_support_set[i][j] >= (self.total_base_classes*2):
                    y_support_set[i][j] = (self.total_base_classes+1) + ( (y_support_set[i][j] - (self.total_base_classes+1)) * 30 )
                
        for i in np.arange( len(y_target) ):
            for j in np.arange( len(y_target[i]) ):
                if y_target[i][j] >= self.total_base_classes and y_target[i][j] < (self.total_base_classes*2):
                    y_target[i][j] = self.total_base_classes + ( (y_target[i][j] - self.total_base_classes) * 30 )
                if y_target[i][j] >= (self.total_base_classes*2):
                    y_target[i][j] = (self.total_base_classes+1) + ( (y_target[i][j] - (self.total_base_classes+1)) * 30 )
        """                    
					
        return x_support_set, y_support_set, x_target, y_target, target_y_actuals        

    def __rotate_data(self, image, k):
        """
        Rotates one image by self.k * 90 degrees counter-clockwise
        :param image: Image to rotate
        :return: Rotated Image
        """
        return np.rot90(image, k)


    def __rotate_batch(self, batch_images, k):
        """
        Rotates a whole image batch
        :param batch_images: A batch of images
        :param k: integer degree of rotation counter-clockwise
        :return: The rotated batch of images
        """
        if not self.is_rotate: 
            return batch_images
        
        batch_size = len(batch_images)
        for i in np.arange(batch_size):
            batch_images[i] = self.__rotate_data(batch_images[i], k)
        return batch_images
