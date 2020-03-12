##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Albert Berenguel
## Computer Vision Center (CVC). Universitat Autonoma de Barcelona
## Email: aberenguel@cvc.uab.es
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import torch.nn as nn
import unittest
import numpy as np
from numpy import array
from models.BidirectionalLSTM import BidirectionalLSTM
from models.Classifier import Classifier
from models.DistanceNetwork import DistanceNetwork
from models.AttentionalClassify import AttentionalClassify
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
import os
import torchvision
import torchvision.transforms as transforms
import json
import math


class MatchingNetwork(nn.Module):
    def __init__(self, keep_prob, \
                 batch_size=100, num_channels=1, learning_rate=0.001, fce=False, num_classes_per_set=5, \
                 num_samples_per_class=1, nClasses = 0, image_size = 28, layer_size = 64, is_use_lstm_layer=False, 
                 vector_dim = None, num_layers=1, dropout=-1, model_path=None, is_use_second_lstm=False):
        super(MatchingNetwork, self).__init__()

        """
        Builds a matching network, the training and evaluation ops as well as data augmentation routines.
        :param keep_prob: A tf placeholder of type tf.float32 denotes the amount of dropout to be used
        :param batch_size: The batch size for the experiment
        :param num_channels: Number of channels of the images
        :param is_training: Flag indicating whether we are training or evaluating
        :param rotate_flag: Flag indicating whether to rotate the images
        :param fce: Flag indicating whether to use full context embeddings (i.e. apply an LSTM on the CNN embeddings)
        :param num_classes_per_set: Integer indicating the number of classes per set
        :param num_samples_per_class: Integer indicating the number of samples per class
        :param nClasses: total number of classes. It changes the output size of the classifier g with a final FC layer.
        :param image_input: size of the input image. It is needed in case we want to create the last FC classification 
        """
        self.is_use_lstm_layer = is_use_lstm_layer
        self.vector_dim = vector_dim
        
        self.batch_size = batch_size
        self.fce = fce
        
        # default `log_dir` is "runs" - we'll be more specific here
        self.is_do_train_logging = True
        if self.is_do_train_logging:
            self.log_interval = 50
            self.last_epoch = -1
            self.batch_index = -1
            self.log_file = os.path.join( os.path.dirname(model_path), 'train_log', 'abslog.json' )
            #self.writer = SummaryWriter( os.path.join( os.path.dirname(model_path), 'train_log' ) )
        
        if not self.is_use_lstm_layer:
            self.g = Classifier(layer_size = layer_size, num_channels=num_channels,
                                nClasses= nClasses, image_size = image_size )
        else:
            if is_use_second_lstm:
                self.g = BidirectionalLSTM(layer_sizes=[layer_size], batch_size=self.batch_size, vector_dim = vector_dim, 
                                            num_layers=num_layers, dropout=dropout, layer_sizes_second_lstm=[int(layer_size/2)], 
                                            batch_size_second_lstm=self.batch_size, vector_dim_second_lstm=layer_size*2, 
                                            num_layers_second_lstm=num_layers, dropout_second_lstm=dropout)
            else:
                self.g = BidirectionalLSTM(layer_sizes=[layer_size], batch_size=self.batch_size, vector_dim = vector_dim, 
                                num_layers=num_layers, dropout=dropout)
                                
        if fce:
            self.lstm = BidirectionalLSTM(layer_sizes=[32], batch_size=self.batch_size, vector_dim = self.g.outSize)
        self.dn = DistanceNetwork()
        self.classify = AttentionalClassify()
        self.keep_prob = keep_prob
        self.num_classes_per_set = num_classes_per_set
        self.num_samples_per_class = num_samples_per_class
        self.learning_rate = learning_rate

    def forward(self, support_set_images, support_set_labels_one_hot, target_image, target_label, is_debug = False, is_evaluation_only = False, y_support_set_org = None, target_y_actuals = None, epoch = -1, support_set_y_actuals = None):
        """
        Builds graph for Matching Networks, produces losses and summary statistics.
        :param support_set_images: A tensor containing the support set images [batch_size, sequence_size, n_channels, 28, 28]
        :param support_set_labels_one_hot: A tensor containing the support set labels [batch_size, sequence_size, n_classes]
        :param target_image: A tensor containing the target image (image to produce label for) [batch_size, n_channels, 28, 28]
        :param target_label: A tensor containing the target label [batch_size, 1]
        :return: 
        """
        # produce embeddings for support set images
        encoded_images = []
        if is_evaluation_only == False:
            for i in np.arange(support_set_images.size(1)):
                if not self.is_use_lstm_layer:
                    gen_encode = self.g(support_set_images[:,i,:,:,:])
                else: 
                    gen_encode, _, _ = self.g( support_set_images[:,i,:,:,:].reshape( support_set_images.shape[0], 1, self.vector_dim ) )
                    gen_encode = gen_encode.reshape( gen_encode.shape[0], gen_encode.shape[2] )
                    
                encoded_images.append(gen_encode)

        log_file_encoded = None
        log_file_similarities = None
        if self.is_do_train_logging and np.mod(epoch, self.log_interval) == 0:
            if self.last_epoch == -1 or not self.last_epoch == epoch:
                self.last_epoch = epoch
                self.batch_index = -1
        
            self.batch_index += 1
        
            log_file = self.log_file + "_epoch-"+str(epoch)+"-"+str(self.batch_index)+".json"
            log_file_actuals = log_file + "_actuals.json"
            log_file_encoded = log_file + "_encoded.json"
            log_file_similarities = log_file + "_similarities.json"

            # load logged array and append and save
            try:
                logs = array( json.load( open( log_file ) ) ) 
                print( "support_set_images ", support_set_images.shape, target_image.shape )
                logs = np.concatenate( ( logs, np.concatenate( ( support_set_images.cpu().detach().numpy(), target_image.cpu().detach().numpy() ), axis=1 ) ), axis=0 )
            except Exception as e:
                print( "support_set_images ", support_set_images.shape, target_image.shape )
                logs = np.concatenate( ( support_set_images.cpu().detach().numpy(), target_image.cpu().detach().numpy() ), axis=1 )
                print( logs.shape )
                
            with open( log_file, 'w') as outfile:
                json.dump( logs.tolist(), outfile) 

            # load logged array and append and save
            try:
                logs = array( json.load( open( log_file_actuals ) ) ) 
                print( "support_set_y ", support_set_y_actuals.shape, target_y_actuals.shape )
                logs = np.concatenate( ( logs, np.concatenate( ( support_set_y_actuals, target_y_actuals ), axis=1 ) ), axis=0 )
            except Exception as e:
                print( "support_set_y ", support_set_y_actuals.shape, target_y_actuals.shape )
                logs = np.concatenate( ( support_set_y_actuals, target_y_actuals ), axis=1 )
                print( logs.shape )
                
            with open( log_file_actuals, 'w') as outfile:
                json.dump( logs.tolist(), outfile) 
        
            ## get some random training images

            ## create grid of images
            #img_grid = torchvision.utils.make_grid(encoded_images)    #(images)

            ## show images
            #matplotlib_imshow(img_grid, one_channel=True)

            ## write to tensorboard
            #self.writer.add_image('support_set_images encoded ', img_grid)
                
        pred_indices = []
        # produce embeddings for target images
        for i in np.arange(target_image.size(1)):
            if not self.is_use_lstm_layer:
                gen_encode = self.g(target_image[:,i,:,:,:])
            else:
                gen_encode, _, _ = self.g(target_image[:,i,:,:,:].reshape( target_image.shape[0], 1, self.vector_dim ))
                gen_encode = gen_encode.reshape( gen_encode.shape[0], gen_encode.shape[2] )
                
            if self.is_do_train_logging and np.mod(epoch, self.log_interval) == 0:
                # load logged array and append and save
                try:
                    logs = array( json.load( open( log_file_encoded ) ) ) 
                    for ei in range(0, len(encoded_images)):
                        t1 = np.array( encoded_images[ei].cpu().detach().numpy() )
                        t2 = np.array( gen_encode[ei].cpu().detach().numpy() )
                        t2 = t2.reshape( (1, t2.shape[0]) )
                        print( "encoded_images ", t1.shape, t2.shape )
                        logs = np.concatenate( ( logs, np.concatenate( ( t1, t2 ), axis=0 ) ), axis=0 )
                        print( logs.shape )
                except Exception as e:
                    is_first = True
                    for ei in range(0, len(encoded_images)):
                        t1 = np.array( encoded_images[ei].cpu().detach().numpy() )
                        t2 = np.array( gen_encode[ei].cpu().detach().numpy() )
                        t2 = t2.reshape( (1, t2.shape[0]) )
                        print( "encoded_images ", t1.shape, t2.shape )
                        if not is_first:
                            logs = np.concatenate( ( logs, np.concatenate( ( t1, t2 ), axis=0 ) ), axis=0 )
                        else:
                            is_first = False
                            logs = np.concatenate( ( t1, t2 ), axis=0 )
                        print( logs.shape )
                    
                with open( log_file_encoded, 'w') as outfile:
                    json.dump( logs.tolist(), outfile) 


                ## get some random training images

                ## create grid of images
                #img_grid = torchvision.utils.make_grid(gen_encode)    #(images)

                ## show images
                #matplotlib_imshow(img_grid, one_channel=True)

                ## write to tensorboard
                #self.writer.add_image('target_image encoded ', img_grid)
                
            #print("gen_encode ", gen_encode.shape)
            encoded_images.append(gen_encode)
            outputs = torch.stack(encoded_images)
            #print("outputs ", outputs.shape)

            if self.fce:
                outputs, hn, cn = self.lstm(outputs)

            # get similarity between support set embeddings and target
            if is_evaluation_only == False:
                similarities = self.dn(support_set=outputs[:-1], input_image=outputs[-1])
            else:
                similarities = self.dn(input_image=outputs[:])
            similarities = similarities.t()

            if self.is_do_train_logging and np.mod(epoch, self.log_interval) == 0:
                # load logged array and append and save
                try:
                    logs = array( json.load( open( log_file_similarities ) ) ) 
                    logs = np.concatenate( ( logs, np.array( similarities.cpu().detach().numpy() ) ), axis=0 )
                except Exception as e:
                    logs = np.array( similarities.cpu().detach().numpy() )
                    
                #save
                with open( log_file_similarities, 'w') as outfile:
                    json.dump( logs.tolist(), outfile) 

                ## get some random training images

                ## create grid of images
                #img_grid = torchvision.utils.make_grid(similarities)    #(images)

                ## show images
                #matplotlib_imshow(img_grid, one_channel=True)

                ## write to tensorboard
                #self.writer.add_image('similarities ', img_grid)
                
            
            # produce predictions for target probabilities
            if is_evaluation_only == False:
                preds = self.classify(similarities,support_set_y=support_set_labels_one_hot)
            else:
                preds = self.classify(similarities)

            # calculate accuracy and crossentropy loss
            values, indices = preds.max(1)
            pred_indices.append( indices )
            if is_debug:
                print( "support set while in predictions debug mode" )
                #print( y_support_set_org )
                print( target_y_actuals[:,i] )
                print( "predictions debug mode" )
                print( values )
                print( indices.squeeze() )
                print( target_label[:,i] )
                
                if False and torch.mean((indices.squeeze() == target_label[:,i]).float()) >= 0.9:
                    print( "accuracy found above limitttttttttttttttttttttttttttttttttttttttt " + str( torch.mean((indices.squeeze() == target_label[:,i]).float()) ) )
                    print( preds )
                
                if F.cross_entropy(preds, target_label[:,i].long()) <= 1.1:
                    print( ".................loss found below limitttttttttttttttttttttttttttttttttttttttt " + str(F.cross_entropy(preds, target_label[:,i].long())))
                    print( preds )
            
            if i == 0:
                accuracy = torch.mean((indices.squeeze() == target_label[:,i]).float())
                crossentropy_loss = F.cross_entropy(preds, target_label[:,i].long())
            else:
                accuracy = accuracy + torch.mean((indices.squeeze() == target_label[:, i]).float())
                crossentropy_loss = crossentropy_loss + F.cross_entropy(preds, target_label[:, i].long())

            # delete the last target image encoding of encoded_images
            encoded_images.pop()

        #if is_debug:
        #    dfsdfsdfsdf
            
        return accuracy/target_image.size(1), crossentropy_loss/target_image.size(1), pred_indices

    def predict(self, support_set_images, support_set_labels_one_hot, target_image, target_label, is_debug = False, is_evaluation_only = False, y_support_set_org = None, target_y_actuals = None):
        """
        Builds graph for Matching Networks, produces losses and summary statistics.
        :param support_set_images: A tensor containing the support set images [batch_size, sequence_size, n_channels, 28, 28]
        :param support_set_labels_one_hot: A tensor containing the support set labels [batch_size, sequence_size, n_classes]
        :param target_image: A tensor containing the target image (image to produce label for) [batch_size, n_channels, 28, 28]
        :param target_label: A tensor containing the target label [batch_size, 1]
        :return: 
        """
        
        target_image_org = np.copy(target_image)
        #target_image = target_image.cuda()
        
        #target_image
        
        # produce embeddings for support set images
        #print( "encoded_images" )
        #print( type(support_set_labels_one_hot) )
        support_set_labels_one_hot_org_shape = support_set_labels_one_hot.shape
        support_set_images_shape_1_ = support_set_images.shape[1]
        #print( support_set_labels_one_hot.shape )
        #print( target_image.shape )
        tmp_one_hot = np.zeros( (support_set_labels_one_hot_org_shape[0], support_set_labels_one_hot_org_shape[1], support_set_labels_one_hot_org_shape[1]) )
        #print( tmp_one_hot.shape )
        support_set_labels_one_hot = []# np.zeros( (target_image.shape[0], target_image.shape[1], target_image.shape[0]) )
        encoded_images = []
        if is_evaluation_only == False:
            import math
            from torch.autograd import Variable
            import itertools
                    
            tot_ec = 0
            tot_emc = 0
            tot_emcll = 0
            tot_emclvl = 0
            uniq_cls = []
            
            emcllcls = []
            emclvlcls = []
            emcllclsl = []
            emclvlclsl = []
            
            emcllcls_n1 = []
            emclvlcls_n1 = []
            emcllclsl_n1 = []
            emclvlclsl_n1 = []
            
            open_match_cnt = {}
            open_match_tot = {}
            open_match_mpr = {}
            
            #TODO tmp.
            tmp_test_cnt = {}
            
            #for i in np.arange(support_set_images.shape[1]):
            for nardr in range(0, 1):   # 2):
            
                if nardr == 0:
                    if support_set_images.shape[0] > target_image.shape[0]:
                        pindstmp = np.random.permutation( support_set_images.shape[0] - np.mod(support_set_images.shape[0],target_image.shape[0])  )
                        #repeat 5 times
                        pinds = np.concatenate( ( pindstmp,  pindstmp ), axis=0 )
                        for rpt in range(0, support_set_labels_one_hot_org_shape[1]-2):
                            pinds = np.concatenate( ( pinds, pindstmp ), axis=0 )
                        
                    else:
                        if support_set_images.shape[0] < target_image.shape[0]:
                            pindstmp = np.concatenate( ( np.random.permutation( support_set_images.shape[0] ), np.random.choice( support_set_images.shape[0], target_image.shape[0] - support_set_images.shape[0] ) ), axis=0 ) 
                            
                            if is_debug:
                                print( "pindstmp", pindstmp )
                                
                            #repeat 5 times
                            tmp_copy = np.copy( pindstmp )
                            if is_debug:
                                print( "tmp_copy 1 ", tmp_copy )
                            np.random.shuffle( tmp_copy )
                            if is_debug:
                                print( "tmp_copy 1 ", tmp_copy )
                            pinds = np.concatenate( ( pindstmp,  tmp_copy ), axis=0 )
                            for rpt in range(0, support_set_labels_one_hot_org_shape[1]-2):
                                tmp_copy = np.copy( pindstmp )
                                if is_debug:
                                    print( "tmp_copy ", rpt+2, tmp_copy )
                                np.random.shuffle( tmp_copy )
                                if is_debug:
                                    print( "tmp_copy ", rpt+2, tmp_copy )
                                pinds = np.concatenate( ( pinds, tmp_copy ), axis=0 )
                        else:
                            pinds = np.arange( support_set_images.shape[0] ) 
                            
                            if is_debug:
                                print( "pindstmp", pinds )
                            #repeat 5 times
                            for rpt in range(0, support_set_labels_one_hot_org_shape[1]-1):
                                pindstmp = np.concatenate( ( np.arange( rpt+1, support_set_images.shape[0] ), np.arange( 0 , rpt+1 ) ), axis=0 )
                                if is_debug:
                                    print( "rpt ", rpt, pindstmp )
                                pinds = np.concatenate( ( pinds, pindstmp ), axis=0 )
                else:
                    uniq_cls = np.array(uniq_cls)
                    pindstmp = np.random.permutation( len(uniq_cls) - np.mod(len(uniq_cls),target_image.shape[0]) )
                    #repeat 5 times
                    raise Exception("NotImplementedError")
                    pinds = np.concatenate( ( pindstmp, pindstmp ), axis=0 )
                    pinds = np.concatenate( ( pinds, pindstmp ), axis=0 )
                    pinds = np.concatenate( ( pinds, pindstmp ), axis=0 )
                    pinds = np.concatenate( ( pinds, pindstmp ), axis=0 )
                    
                for tatmpts in range(0, target_image.shape[0]):
                        
                    #diagonal 
                    tatmpts_diag = -1
                    if tatmpts == 0:
                        tatmpts_diag = 0
                    elif tatmpts == 1:
                        tatmpts_diag = 4                        
                    elif tatmpts == 2:
                        tatmpts_diag = 1                        
                    elif tatmpts == 3:
                        tatmpts_diag = 5                        
                    elif tatmpts == 4:
                        tatmpts_diag = 2                        
                    elif tatmpts == 5:
                        tatmpts_diag = 6                        
                    elif tatmpts == 6:
                        tatmpts_diag = 3                        
                        
                    if not tatmpts in tmp_test_cnt:
                        tmp_test_cnt[tatmpts] = []
                        
                    if not tatmpts in open_match_cnt:
                        open_match_cnt[tatmpts] = 0
                        open_match_tot[tatmpts] = 0.0
                        open_match_mpr[tatmpts] = 0.0
                        
                    #pindsjj_tmp = np.random.permutation( support_set_images.shape[1] )
                    pindsjj_tmp = np.random.permutation( support_set_images_shape_1_ )
                    #repeat 5 times
                    pindsjj = np.concatenate( ( pindsjj_tmp,  pindsjj_tmp ), axis=0 )
                    for rpt in range(0, support_set_labels_one_hot_org_shape[1]-2):
                        pindsjj = np.concatenate( ( pindsjj, pindsjj_tmp ), axis=0 )
                        
                    jjcntr = 0
                    for jj in range( 0, support_set_images_shape_1_):   #support_set_images.shape[1] ):   #int( math.floor(support_set_images.shape[1] / support_set_labels_one_hot_org_shape[1]) ) ): 
                    
                        ii_cntr = 0
                        tstcls = 0
                        iilength = int( math.floor( support_set_images.shape[0] / target_image.shape[0] ) ) if nardr == 0 else int( math.floor( len(uniq_cls) / target_image.shape[0] ) )
                        iilength = 1 if iilength == 0 else iilength
                        for ii in range( 0, iilength ): 
                            encoded_images = []
                            
                            if is_debug:
                                print( "tatmpts " + str(tatmpts) +" jj " + str(jj) + "  ii " + str(ii) )
                                
                            xhat_pinds = np.concatenate( ( np.random.permutation( support_set_labels_one_hot_org_shape[1] ), np.random.choice( support_set_labels_one_hot_org_shape[1], target_image.shape[0] - support_set_labels_one_hot_org_shape[1] ) ), axis=0 ) #np.random.choice( support_set_labels_one_hot_org_shape[1], target_image.shape[0] ) #np.random.permutation( support_set_labels_one_hot_org_shape[1] )
                            if is_debug:
                                print( xhat_pinds )
                                
                            xhat_ind = 0
                            for j in range(0, support_set_labels_one_hot_org_shape[1]):
                            
                                jinds = pindsjj[jjcntr]  #int( math.floor(ii_cntr/iilength) ) + (jj*support_set_labels_one_hot_org_shape[1])  #( j )+(jj*support_set_labels_one_hot_org_shape[1])  
                                jjcntr += 1 
                                if tatmpts_diag == xhat_ind:
                                    tmp_test_cnt[tatmpts].append( jinds )
                                
                                #print( "tatmpts " + str(tatmpts) + " j " + str(j) + " jinds " + str(jinds) )
                                
                                #print( support_set_images[pinds[ii_cntr*target_image.shape[0]:(ii_cntr+1)*target_image.shape[0]],j+(jj*support_set_labels_one_hot_org_shape[1]),:,:,:].shape )
                                if nardr == 0:
                                    if not self.is_use_lstm_layer:
                                        gen_encode = self.g( torch.Tensor(support_set_images[pinds[ii_cntr*target_image.shape[0]:(ii_cntr+1)*target_image.shape[0]],jinds,:,:,:]) )
                                    else: 
                                        if torch.cuda.is_available():
                                            gen_encode, _, _ = self.g( torch.Tensor(support_set_images[pinds[ii_cntr*target_image.shape[0]:(ii_cntr+1)*target_image.shape[0]],jinds,:,:,:].reshape( target_image.shape[0], 1, self.vector_dim ) ).cuda() )
                                        else:
                                            gen_encode, _, _ = self.g( torch.Tensor(support_set_images[pinds[ii_cntr*target_image.shape[0]:(ii_cntr+1)*target_image.shape[0]],jinds,:,:,:].reshape( target_image.shape[0], 1, self.vector_dim ) ) )
                                        gen_encode = gen_encode.reshape( gen_encode.shape[0], gen_encode.shape[2] )
                                else:
                                    gen_encode = self.g( torch.Tensor(support_set_images[uniq_cls[pinds[ii_cntr*target_image.shape[0]:(ii_cntr+1)*target_image.shape[0]]],jinds,:,:,:]) )
                                #print( gen_encode.shape )
                                
                                encoded_images.append( Variable(gen_encode, volatile=True).float() )
                                tmp_one_hot[:,j,j] = 1

                                #prepare target
                                n_test_samples = np.sum(j == xhat_pinds)
                                for xhat_i in range(0, n_test_samples):
                                    if not tatmpts_diag == xhat_ind:
                                        if nardr == 0:
                                            if torch.cuda.is_available():
                                                #target_image[xhat_ind,0,:,:,:] = Variable(torch.from_numpy(support_set_images[pinds[(ii_cntr*target_image.shape[0])+xhat_ind:(ii_cntr*target_image.shape[0])+xhat_ind+1],np.random.randint(0, support_set_images.shape[1]-2),:,:,:]), volatile=True).float().cuda()
                                                target_image[xhat_ind,0,:,:,:] = Variable(torch.from_numpy(support_set_images[pinds[(ii_cntr*target_image.shape[0])+xhat_ind:(ii_cntr*target_image.shape[0])+xhat_ind+1],np.random.randint(0, support_set_images_shape_1_-2),:,:,:]), volatile=True).float().cuda()
                                            else:
                                                #target_image[xhat_ind,0,:,:,:] = Variable(torch.from_numpy(support_set_images[pinds[(ii_cntr*target_image.shape[0])+xhat_ind:(ii_cntr*target_image.shape[0])+xhat_ind+1],np.random.randint(0, support_set_images.shape[1]-2),:,:,:]), volatile=True).float()
                                                target_image[xhat_ind,0,:,:,:] = Variable(torch.from_numpy(support_set_images[pinds[(ii_cntr*target_image.shape[0])+xhat_ind:(ii_cntr*target_image.shape[0])+xhat_ind+1],np.random.randint(0, support_set_images_shape_1_-2),:,:,:]), volatile=True).float()
                                        else:
                                            #target_image[xhat_ind,0,:,:,:] = Variable(torch.from_numpy(support_set_images[uniq_cls[pinds[(ii_cntr*target_image.shape[0])+xhat_ind:(ii_cntr*target_image.shape[0])+xhat_ind+1]],np.random.randint(0, support_set_images.shape[1]-2),:,:,:]), volatile=True).float()
                                            target_image[xhat_ind,0,:,:,:] = Variable(torch.from_numpy(support_set_images[uniq_cls[pinds[(ii_cntr*target_image.shape[0])+xhat_ind:(ii_cntr*target_image.shape[0])+xhat_ind+1]],np.random.randint(0, support_set_images_shape_1_-2),:,:,:]), volatile=True).float()
                                        #target_image[xhat_ind,0,:,:,:] = Variable(torch.from_numpy(support_set_images[pinds[(ii_cntr*target_image.shape[0])+xhat_ind:(ii_cntr*target_image.shape[0])+xhat_ind+1],j+(jj*support_set_labels_one_hot_org_shape[1]),:,:,:]), volatile=True).float()
                                    else:
                                        if nardr == 0:
                                            tstcls = pinds[(ii_cntr*target_image.shape[0])+xhat_ind:(ii_cntr*target_image.shape[0])+xhat_ind+1]
                                        else:
                                            tstcls = uniq_cls[pinds[(ii_cntr*target_image.shape[0])+xhat_ind:(ii_cntr*target_image.shape[0])+xhat_ind+1]]
                                            
                                        if torch.cuda.is_available():
                                            target_image[xhat_ind,0,:,:,:] = Variable(torch.from_numpy(target_image_org[xhat_ind,0,:,:,:]), volatile=True).float().cuda()
                                        else:
                                            target_image[xhat_ind,0,:,:,:] = Variable(torch.from_numpy(target_image_org[xhat_ind,0,:,:,:]), volatile=True).float()
                                    target_label[xhat_ind,0] = Variable( torch.from_numpy( np.array( [j] ) ), volatile=True).long()
                                    xhat_ind = xhat_ind + 1 
                                
                                ii_cntr = ii_cntr + 1

                            
                            """                    
                            #randcls = np.random.randint(0, target_image.shape[0])
                            pjs = np.random.permutation( target_image.shape[0] )
                            print("pjs")
                            print(pjs)
                            target_image[:,0,:,:,:] = Variable(torch.from_numpy(support_set_images[pinds[ii*target_image.shape[0]:(ii+1)*target_image.shape[0]],pjs,:,:,:]), volatile=True).float()
                            target_label[:,0] = Variable(torch.from_numpy( np.random.choice( support_set_labels_one_hot_org_shape[1], target_image.shape[0] ) ), volatile=True).long()
                            print(target_label)
                            """
                                
                            """
                            pinds = np.random.permutation( gen_encode.shape[0] - np.mod(gen_encode.shape[0],target_image.shape[0])  )
                            for gei in range( 0, int( math.floor(gen_encode.shape[0] / target_image.shape[0]) ) ): 
                                encoded_images.append( gen_encode[ pinds[gei*target_image.shape[0]:(gei+1)*target_image.shape[0]], :] )
                                
                                for ci in range(0, target_image.shape[1]):
                                    support_set_labels_one_hot.append( tmp_one_hot )
                                
                                support_set_labels_one_hot.append( tmp_one_hot )
                                
                                if (gei+2)*target_image.shape[0] >= gen_encode.shape[0]:
                                    break
                            """
                            
                            #print("tmp_one_hot")
                            #print(tmp_one_hot.shape)
                            support_set_labels_one_hot = tmp_one_hot
                            #break
                                
                            support_set_labels_one_hot = Variable(torch.from_numpy(support_set_labels_one_hot), volatile=True).float()
                            #print( type(support_set_labels_one_hot) )
                            #print( support_set_labels_one_hot.shape )
                                
                            tot_ec = tot_ec + 1
                            pred_indices = []
                    # produce embeddings for target images
                    #for i in np.arange(target_image.size(1)):
                            i = 0
                            #print( "target gen_encode" )
                            #print( target_image[:,i,:,:,:].shape )
                            if not self.is_use_lstm_layer:
                                gen_encode = self.g(target_image[:,i,:,:,:])
                            else: 
                                if torch.cuda.is_available():
                                    gen_encode, _, _ = self.g( torch.Tensor(target_image[:,i,:,:,:].reshape( target_image.shape[0], 1, self.vector_dim )).cuda() )
                                else:
                                    gen_encode, _, _ = self.g( torch.Tensor(target_image[:,i,:,:,:].reshape( target_image.shape[0], 1, self.vector_dim )) )
                                gen_encode = gen_encode.reshape( gen_encode.shape[0], gen_encode.shape[2] )
                            #print( gen_encode.shape )
                            
                            #encoded_images.append(gen_encode)
                            #print( type(encoded_images) )
                            #outputs = torch.stack(encoded_images)
                            #print( type(outputs) )
                            #print( outputs.shape )
                            

                            if self.fce:
                                raise Exception("NotImplementedError, outputs does not contain target image")
                                outputs, hn, cn = self.lstm(outputs)

                            # get similarity between support set embeddings and target
                            if is_evaluation_only == False:
                                #similarities = self.dn(support_set=outputs[:-1], input_image=outputs[-1])
                                similarities = self.dn(support_set=encoded_images, input_image=gen_encode)
                            else:
                                similarities = self.dn(input_image=outputs[:])
                            similarities = similarities.t()

                            # produce predictions for target probabilities
                            if is_evaluation_only == False:
                                if torch.cuda.is_available():
                                    preds = self.classify(similarities,support_set_y=support_set_labels_one_hot.cuda())
                                else:
                                    preds = self.classify(similarities,support_set_y=support_set_labels_one_hot)
                            else:
                                preds = self.classify(similarities)

                            #print(support_set_labels_one_hot)
                            #print(preds)
                                
                            # calculate accuracy and crossentropy loss
                            values, indices = preds.max(1)
                            pred_indices.append( indices )
                            
                            if indices.squeeze()[tatmpts_diag] == target_label[tatmpts_diag,i]:
                                open_match_cnt[tatmpts] += 1
                                open_match_tot[tatmpts] += values[tatmpts]
                                open_match_mpr[tatmpts] = open_match_tot[tatmpts] / open_match_cnt[tatmpts]
                            
                            if is_debug:
                                #print( "support set while in predictions debug mode" )
                                ##print( y_support_set_org )
                                #print( target_y_actuals[:,i] )
                                if True or nardr >= 1:
                                    print( "predictions debug mode" )
                                    print( values )
                                    print( indices.squeeze() )
                                    print( target_label[:,i] )
                                    print( ".................loss found below limitttttttttttttttttttttttttttttttttttttttt " + str(F.cross_entropy(preds, target_label[:,i].long())))
                                    print( "accuracy found above limitttttttttttttttttttttttttttttttttttttttt " + str( torch.mean((indices.squeeze() == target_label[:,i]).float()) ) )
                            
                                if False and torch.mean((indices.squeeze() == target_label[:,i]).float()) >= 0.9:
                                    print( "accuracy found above limitttttttttttttttttttttttttttttttttttttttt " + str( torch.mean((indices.squeeze() == target_label[:,i]).float()) ) )
                                    print( preds )
                                
                                #if (F.cross_entropy(preds, target_label[:,i].long()) <= 1.25 and values[tatmpts] >= 0.75) or nardr >= 1:
                                if (F.cross_entropy(preds, target_label[:,i].long()) <= 1.35 and torch.mean((indices.squeeze() == target_label[:,i]).float()) >= 1.0) or nardr >= 1:
                                    print( ".................loss found below limitttttttttttttttttttttttttttttttttttttttt " + str(F.cross_entropy(preds, target_label[:,i].long())))
                                    print( preds )
                                    tot_emc = tot_emc + 1
                                    #if (F.cross_entropy(preds, target_label[:,i].long()) <= 1.25 and values[tatmpts] >= 0.98):
                                    if (F.cross_entropy(preds, target_label[:,i].long()) <= 1.35 and torch.mean((indices.squeeze() == target_label[:,i]).float()) >= 1.0):
                                        tot_emcll = tot_emcll + 1
                                        
                                        #if F.cross_entropy(preds, target_label[:,i].long()) <= 1.19 and values[tatmpts] >= 0.99:
                                        if (F.cross_entropy(preds, target_label[:,i].long()) <= 1.35 and torch.mean((indices.squeeze() == target_label[:,i]).float()) >= 1.0):
                                            tot_emclvl = tot_emclvl + 1
                                            if nardr == 0:
                                                if not tstcls[0] in uniq_cls:
                                                    uniq_cls.append(tstcls[0])
                                            
                                                emclvlcls.append( tstcls[0] )
                                                #emclvlclsl.append( F.cross_entropy(preds, target_label[:,i].long()) )
                                                emclvlclsl.append( values[tatmpts] )
                                            else:
                                                emclvlcls_n1.append( tstcls[0] )
                                                #emclvlclsl.append( F.cross_entropy(preds, target_label[:,i].long()) )
                                                emclvlclsl_n1.append( values[tatmpts] )
                                        elif values[tatmpts] >= 0.985:
                                            if nardr == 0:
                                                if not tstcls[0] in uniq_cls:
                                                    uniq_cls.append(tstcls[0])
                                                        
                                                emcllcls.append( tstcls[0] )
                                                #emcllclsl.append( F.cross_entropy(preds, target_label[:,i].long()) )
                                                emcllclsl.append( values[tatmpts] )
                                            else:
                                                emcllcls_n1.append( tstcls[0] )
                                                #emcllclsl.append( F.cross_entropy(preds, target_label[:,i].long()) )
                                                emcllclsl_n1.append( values[tatmpts] )

                                
                            if i == 0:
                                accuracy = torch.mean((indices.squeeze() == target_label[:,i]).float())
                                crossentropy_loss = F.cross_entropy(preds, target_label[:,i].long())
                            else:
                                accuracy = accuracy + torch.mean((indices.squeeze() == target_label[:, i]).float())
                                crossentropy_loss = crossentropy_loss + F.cross_entropy(preds, target_label[:, i].long())
                            
                            #if is_debug:                    
                                #print( "test_loss: {}, test_accuracy: {}".format(crossentropy_loss.data, accuracy.data) )

                            ## delete the last target image encoding of encoded_images
                            #encoded_images.pop()
                            
                            #return accuracy, crossentropy_loss, pred_indices

        #if is_debug:
        #    dfsdfsdfsdf
            
                if is_debug:
                    print( "tmp_test_cnt", tmp_test_cnt )
                    print( "open_match_cnt", open_match_cnt )
                    print( "open_match_mpr", open_match_mpr )
                
                    print( "tot_ec " + str(tot_ec) + " tot_emc " + str(tot_emc) + " tot_emcll " + str(tot_emcll) + " tot_emclvl " + str(tot_emclvl) )
                    print( "emcllcls ", emcllcls )
                    print( "emcllclsl ", emcllclsl )
                    print( "emclvlcls ", emclvlcls )
                    print( "emclvlclsl ", emclvlclsl )
                    
                    print( "emcllcls_n1 ", emcllcls_n1 )
                    print( "emcllclsl_n1 ", emcllclsl_n1 )
                    print( "emclvlcls_n1 ", emclvlcls_n1 )
                    print( "emclvlclsl_n1 ", emclvlclsl_n1 )
            
        return accuracy/target_image.size(1), crossentropy_loss/target_image.size(1), pred_indices, emcllcls, emcllclsl, emclvlcls, emclvlclsl, open_match_cnt, open_match_mpr
        
class MatchingNetworkTest(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_accuracy(self):
        pass


if __name__ == '__main__':
    unittest.main()



