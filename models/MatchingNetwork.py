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
                 num_samples_per_class=1, nClasses = 0, image_size = 28):
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
        self.batch_size = batch_size
        self.fce = fce
        self.g = Classifier(layer_size = 64, num_channels=num_channels,
                            nClasses= nClasses, image_size = image_size )
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
                gen_encode = self.g(support_set_images[:,i,:,:,:])
                encoded_images.append(gen_encode)

        pred_indices = []
        # produce embeddings for target images
        for i in np.arange(target_image.size(1)):
            gen_encode = self.g(target_image[:,i,:,:,:])
            encoded_images.append(gen_encode)
            outputs = torch.stack(encoded_images)

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
        # produce embeddings for support set images
        print( "encoded_images" )
        print( type(support_set_labels_one_hot) )
        print( support_set_labels_one_hot.shape )
        support_set_labels_one_hot = np.zeros(support_set_images.shape[0], support_set_images.shape[1])
        encoded_images = []
        if is_evaluation_only == False:
            for i in np.arange(support_set_images.shape[1]):
                gen_encode = self.g( torch.Tensor(support_set_images[:,i,:,:,:]) )
                encoded_images.append(gen_encode)
                for ci in range(0, support_set_images.shape[0]):
                    support_set_labels_one_hot[ci, i] = ci
        
        support_set_labels_one_hot = np.array(support_set_labels_one_hot)
        print( type(support_set_labels_one_hot) )
        print( support_set_labels_one_hot.shape )
                    
                   
        pred_indices = []
        # produce embeddings for target images
        for i in np.arange(target_image.size(1)):
            gen_encode = self.g(target_image[:,i,:,:,:])
            encoded_images.append(gen_encode)
            print( type(encoded_images) )
            print( encoded_images.size() )
            outputs = torch.stack(encoded_images)
            print( type(outputs) )
            print( outputs.shape )
            

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
            
            return accuracy, crossentropy_loss, pred_indices

        #if is_debug:
        #    dfsdfsdfsdf
            
        return accuracy/target_image.size(1), crossentropy_loss/target_image.size(1), pred_indices
        
class MatchingNetworkTest(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_accuracy(self):
        pass


if __name__ == '__main__':
    unittest.main()



