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
from models.BidirectionalLSTM import BidirectionalLSTM
from models.Classifier import Classifier
from models.DistanceNetwork import DistanceNetwork
from models.AttentionalClassify import AttentionalClassify
import torch.nn.functional as F

class MatchingNetwork(nn.Module):
    def __init__(self, keep_prob, \
                 batch_size=100, num_channels=1, learning_rate=0.001, fce=False, num_classes_per_set=5, \
                 num_samples_per_class=1, nClasses = 0, image_size = 28, layer_size = 64, is_use_lstm_layer=False, vector_dim = None):
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
        if not self.is_use_lstm_layer:
            self.g = Classifier(layer_size = layer_size, num_channels=num_channels,
                                nClasses= nClasses, image_size = image_size )
        else:
            self.g = BidirectionalLSTM(layer_sizes=[layer_size], batch_size=self.batch_size, vector_dim = vector_dim)
                                
        if fce:
            self.lstm = BidirectionalLSTM(layer_sizes=[32], batch_size=self.batch_size, vector_dim = self.g.outSize)
        self.dn = DistanceNetwork()
        self.classify = AttentionalClassify()
        self.keep_prob = keep_prob
        self.num_classes_per_set = num_classes_per_set
        self.num_samples_per_class = num_samples_per_class
        self.learning_rate = learning_rate

    def forward(self, support_set_images, support_set_labels_one_hot, target_image, target_label, is_debug = False, is_evaluation_only = False, y_support_set_org = None, target_y_actuals = None):
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

        pred_indices = []
        # produce embeddings for target images
        for i in np.arange(target_image.size(1)):
            if not self.is_use_lstm_layer:
                gen_encode = self.g(target_image[:,i,:,:,:])
            else:
                gen_encode, _, _ = self.g(target_image[:,i,:,:,:].reshape( target_image.shape[0], 1, self.vector_dim ))
                gen_encode = gen_encode.reshape( gen_encode.shape[0], gen_encode.shape[2] )
                
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
        
        #target_image
        
        # produce embeddings for support set images
        #print( "encoded_images" )
        #print( type(support_set_labels_one_hot) )
        support_set_labels_one_hot_org_shape = support_set_labels_one_hot.shape
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
            #for i in np.arange(support_set_images.shape[1]):
            for nardr in range(0, 1):   # 2):
                if nardr == 0:
                    pindstmp = np.random.permutation( support_set_images.shape[0] - np.mod(support_set_images.shape[0],target_image.shape[0])  )
                    #repeat 5 times
                    pinds = np.concatenate( ( pindstmp, pindstmp ), axis=0 )
                    pinds = np.concatenate( ( pinds, pindstmp ), axis=0 )
                    pinds = np.concatenate( ( pinds, pindstmp ), axis=0 )
                    pinds = np.concatenate( ( pinds, pindstmp ), axis=0 )
                else:
                    uniq_cls = np.array(uniq_cls)
                    pindstmp = np.random.permutation( len(uniq_cls) - np.mod(len(uniq_cls),target_image.shape[0]) )
                    #repeat 5 times
                    pinds = np.concatenate( ( pindstmp, pindstmp ), axis=0 )
                    pinds = np.concatenate( ( pinds, pindstmp ), axis=0 )
                    pinds = np.concatenate( ( pinds, pindstmp ), axis=0 )
                    pinds = np.concatenate( ( pinds, pindstmp ), axis=0 )
                    
                for tatmpts in range(0, target_image.shape[0]):
                        
                    for jj in range( 0, int( math.floor(support_set_images.shape[1] / support_set_labels_one_hot_org_shape[1]) ) ): 
                        ii_cntr = 0
                        tstcls = 0
                        iilength = int( math.floor( support_set_images.shape[0] / target_image.shape[0] ) ) if nardr == 0 else int( math.floor( len(uniq_cls) / target_image.shape[0] ) )
                        for ii in range( 0, iilength ): 
                            encoded_images = []
                            
                            #print( "gen_encode jj " + str(jj) + "  ii " + str(ii) )
                            xhat_pinds = np.concatenate( ( np.random.permutation( support_set_labels_one_hot_org_shape[1] ), np.random.choice( support_set_labels_one_hot_org_shape[1], target_image.shape[0] - support_set_labels_one_hot_org_shape[1] ) ), axis=0 ) #np.random.choice( support_set_labels_one_hot_org_shape[1], target_image.shape[0] ) #np.random.permutation( support_set_labels_one_hot_org_shape[1] )
                            xhat_ind = 0
                            for j in range(0, support_set_labels_one_hot_org_shape[1]):
                            
                                jinds = int( math.floor(ii_cntr/iilength) ) + (jj*support_set_labels_one_hot_org_shape[1])  #( j )+(jj*support_set_labels_one_hot_org_shape[1])  
                                #print( support_set_images[pinds[ii_cntr*target_image.shape[0]:(ii_cntr+1)*target_image.shape[0]],j+(jj*support_set_labels_one_hot_org_shape[1]),:,:,:].shape )
                                if nardr == 0:
                                    if not self.is_use_lstm_layer:
                                        gen_encode = self.g( torch.Tensor(support_set_images[pinds[ii_cntr*target_image.shape[0]:(ii_cntr+1)*target_image.shape[0]],jinds,:,:,:]) )
                                    else: 
                                        gen_encode, _, _ = self.g( torch.Tensor(support_set_images[pinds[ii_cntr*target_image.shape[0]:(ii_cntr+1)*target_image.shape[0]],jinds,:,:,:].reshape( support_set_images.shape[0], 1, self.vector_dim ).cuda() ) )
                                        gen_encode = gen_encode.reshape( gen_encode.shape[0], gen_encode.shape[2] )
                                else:
                                    gen_encode = self.g( torch.Tensor(support_set_images[uniq_cls[pinds[ii_cntr*target_image.shape[0]:(ii_cntr+1)*target_image.shape[0]]],jinds,:,:,:]) )
                                #print( gen_encode.shape )
                                
                                encoded_images.append( Variable(gen_encode, volatile=True).float() )
                                tmp_one_hot[:,j,j] = 1

                                #prepare target
                                n_test_samples = np.sum(j == xhat_pinds)
                                for xhat_i in range(0, n_test_samples):
                                    if not tatmpts == xhat_ind:
                                        if nardr == 0:
                                            target_image[xhat_ind,0,:,:,:] = Variable(torch.from_numpy(support_set_images[pinds[(ii_cntr*target_image.shape[0])+xhat_ind:(ii_cntr*target_image.shape[0])+xhat_ind+1],np.random.randint(0, support_set_images.shape[1]-2),:,:,:]), volatile=True).float()
                                        else:
                                            target_image[xhat_ind,0,:,:,:] = Variable(torch.from_numpy(support_set_images[uniq_cls[pinds[(ii_cntr*target_image.shape[0])+xhat_ind:(ii_cntr*target_image.shape[0])+xhat_ind+1]],np.random.randint(0, support_set_images.shape[1]-2),:,:,:]), volatile=True).float()
                                        #target_image[xhat_ind,0,:,:,:] = Variable(torch.from_numpy(support_set_images[pinds[(ii_cntr*target_image.shape[0])+xhat_ind:(ii_cntr*target_image.shape[0])+xhat_ind+1],j+(jj*support_set_labels_one_hot_org_shape[1]),:,:,:]), volatile=True).float()
                                    else:
                                        if nardr == 0:
                                            tstcls = pinds[(ii_cntr*target_image.shape[0])+xhat_ind:(ii_cntr*target_image.shape[0])+xhat_ind+1]
                                        else:
                                            tstcls = uniq_cls[pinds[(ii_cntr*target_image.shape[0])+xhat_ind:(ii_cntr*target_image.shape[0])+xhat_ind+1]]
                                            
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
                                gen_encode, _, _ = self.g( torch.Tensor(target_image[:,i,:,:,:].reshape( target_image.shape[0], 1, self.vector_dim )).cuda() )
                                gen_encode = gen_encode.reshape( gen_encode.shape[0], gen_encode.shape[2] )
                            #print( gen_encode.shape )
                            
                            #encoded_images.append(gen_encode)
                            #print( type(encoded_images) )
                            #outputs = torch.stack(encoded_images)
                            #print( type(outputs) )
                            #print( outputs.shape )
                            

                            if self.fce:
                                raise Exception("outputs does not contain target image")
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
                                preds = self.classify(similarities,support_set_y=support_set_labels_one_hot.cuda())
                            else:
                                preds = self.classify(similarities)

                            #print(support_set_labels_one_hot)
                            #print(preds)
                                
                            # calculate accuracy and crossentropy loss
                            values, indices = preds.max(1)
                            pred_indices.append( indices )
                            if is_debug:
                                #print( "support set while in predictions debug mode" )
                                ##print( y_support_set_org )
                                #print( target_y_actuals[:,i] )
                                if nardr >= 1:
                                    print( "predictions debug mode" )
                                    print( values )
                                    print( indices.squeeze() )
                                    print( target_label[:,i] )
                                
                                if False and torch.mean((indices.squeeze() == target_label[:,i]).float()) >= 0.9:
                                    print( "accuracy found above limitttttttttttttttttttttttttttttttttttttttt " + str( torch.mean((indices.squeeze() == target_label[:,i]).float()) ) )
                                    print( preds )
                                
                                if (F.cross_entropy(preds, target_label[:,i].long()) <= 0.95 and values[tatmpts] >= 0.97) or nardr >= 1:
                                    print( ".................loss found below limitttttttttttttttttttttttttttttttttttttttt " + str(F.cross_entropy(preds, target_label[:,i].long())))
                                    print( preds )
                                    tot_emc = tot_emc + 1
                                    if (F.cross_entropy(preds, target_label[:,i].long()) <= 0.95 and values[tatmpts] >= 0.98):
                                        tot_emcll = tot_emcll + 1
                                        
                                        if F.cross_entropy(preds, target_label[:,i].long()) <= 0.93 and values[tatmpts] >= 0.995:
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
            
                print("tot_ec " + str(tot_ec) + " tot_emc " + str(tot_emc) + " tot_emcll " + str(tot_emcll) + " tot_emclvl " + str(tot_emclvl) )
                print( "emcllcls ", emcllcls )
                print( "emcllclsl ", emcllclsl )
                print( "emclvlcls ", emclvlcls )
                print( "emclvlclsl ", emclvlclsl )
                
                print( "emcllcls_n1 ", emcllcls_n1 )
                print( "emcllclsl_n1 ", emcllclsl_n1 )
                print( "emclvlcls_n1 ", emclvlcls_n1 )
                print( "emclvlclsl_n1 ", emclvlclsl_n1 )
            
        return accuracy/target_image.size(1), crossentropy_loss/target_image.size(1), pred_indices, emcllcls, emcllclsl, emclvlcls, emclvlclsl
        
class MatchingNetworkTest(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_accuracy(self):
        pass


if __name__ == '__main__':
    unittest.main()



