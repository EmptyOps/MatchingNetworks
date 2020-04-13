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
import torch.backends.cudnn as cudnn
import tqdm
from models.MatchingNetwork import MatchingNetwork
from torch.autograd import Variable
import os

class OneShotBuilder:

    def __init__(self, data, model_path=""):
        """
        Initializes an OneShotBuilder object. The OneShotBuilder object takes care of setting up our experiment
        and provides helper functions such as run_training_epoch and run_validation_epoch to simplify out training
        and evaluation procedures.
        :param data: A data provider class
        """
        self.data = data
        self.model_path = model_path

    def build_experiment(self, batch_size, classes_per_set, samples_per_class, channels, fce, image_size = 28, layer_size = 1600, 
                            is_use_lstm_layer=False, vector_dim = None, num_layers=1, dropout=-1, is_use_second_lstm = False):

        """
        :param batch_size: The experiment batch size
        :param classes_per_set: An integer indicating the number of classes per support set
        :param samples_per_class: An integer indicating the number of samples per class
        :param channels: The image channels
        :param fce: Whether to use full context embeddings or not
        :return: a matching_network object, along with the losses, the training ops and the init op
        """
        self.classes_per_set = classes_per_set
        self.samples_per_class = samples_per_class
        self.keep_prob = torch.FloatTensor(1)
        self.isCudaAvailable = torch.cuda.is_available()
        if not os.path.exists(self.model_path):
            self.matchingNet = MatchingNetwork(batch_size=batch_size,
                                             keep_prob=self.keep_prob, num_channels=channels,
                                             fce=fce,
                                             num_classes_per_set=classes_per_set,
                                             num_samples_per_class=samples_per_class,
                                             nClasses = 0, image_size = image_size, layer_size = layer_size, 
                                             is_use_lstm_layer=is_use_lstm_layer, vector_dim = vector_dim, 
                                             num_layers=num_layers, dropout=dropout, model_path=self.model_path, 
                                             is_use_second_lstm=is_use_second_lstm)
        else: 
            if self.isCudaAvailable:
                self.matchingNet = torch.load(self.model_path)
            else:
                self.matchingNet = torch.load(self.model_path, map_location=torch.device('cpu'))
                                             
        self.optimizer = 'adam'
        self.lr = 1e-03
        self.current_lr = 1e-03
        self.lr_decay = 1e-6
        self.wd = 1e-4
        self.total_train_iter = 0
        if self.isCudaAvailable:
            cudnn.benchmark = True
            torch.cuda.manual_seed_all(0)
            self.matchingNet.cuda()

    def load_model(self):
        self.matchingNet = torch.load(self.model_path)
            
    def save_model(self, epoch):
        torch.save(self.matchingNet, self.model_path.replace('EPOCH', str(epoch)))
            
    def run_training_epoch(self, total_train_batches, epoch=-1):
        """
        Runs one training epoch
        :param total_train_batches: Number of batches to train on
        :return: mean_training_categorical_crossentropy_loss and mean_training_accuracy
        """
        total_c_loss = 0.
        total_accuracy = 0.
        # Create the optimizer
        optimizer = self.__create_optimizer(self.matchingNet, self.lr)

        with tqdm.tqdm(total=total_train_batches) as pbar:
            for i in range(total_train_batches):  # train epoch
                x_support_set, y_support_set, x_target, y_target, support_set_y_actuals, target_y_actuals = \
                    self.data.get_batch_training(str_type = 'train',rotate_flag = True)

                x_support_set = Variable(torch.from_numpy(x_support_set)).float()
                y_support_set = Variable(torch.from_numpy(y_support_set),requires_grad=False).long()
                x_target = Variable(torch.from_numpy(x_target)).float()
                y_target = Variable(torch.from_numpy(y_target),requires_grad=False).long()

                # y_support_set: Add extra dimension for the one_hot
                y_support_set = torch.unsqueeze(y_support_set, 2)
                sequence_length = y_support_set.size()[1]
                batch_size = y_support_set.size()[0]
                y_support_set_one_hot = torch.FloatTensor(batch_size, sequence_length,
                                                               self.classes_per_set).zero_()
                y_support_set_one_hot.scatter_(2, y_support_set.data, 1)
                y_support_set_one_hot = Variable(y_support_set_one_hot)

                # Reshape channels
                size = x_support_set.size()
                x_support_set = x_support_set.view(size[0],size[1],size[4],size[2],size[3])
                size = x_target.size()
                x_target = x_target.view(size[0],size[1],size[4],size[2],size[3])
                if self.isCudaAvailable:
                    acc, c_loss_value, _ = self.matchingNet(x_support_set.cuda(), y_support_set_one_hot.cuda(),
                                                         x_target.cuda(), y_target.cuda(), epoch = epoch, 
                                                         target_y_actuals = target_y_actuals, 
                                                         support_set_y_actuals = support_set_y_actuals )
                else:
                    acc, c_loss_value, _ = self.matchingNet(x_support_set, y_support_set_one_hot,
                                                         x_target, y_target, epoch = epoch, 
                                                         target_y_actuals = target_y_actuals, 
                                                         support_set_y_actuals = support_set_y_actuals)

                # Before the backward pass, use the optimizer object to zero all of the
                # gradients for the variables it will update (which are the learnable weights
                # of the model)
                optimizer.zero_grad()

                # Backward pass: compute gradient of the loss with respect to model parameters
                c_loss_value.backward()

                # Calling the step function on an Optimizer makes an update to its parameters
                optimizer.step()

                # update the optimizer learning rate
                self.__adjust_learning_rate(optimizer)

                #iter_out = "tr_loss: {}, tr_accuracy: {}".format(c_loss_value.data[0], acc.data[0])
                iter_out = "tr_loss: {}, tr_accuracy: {}".format(c_loss_value.data, acc.data)
                pbar.set_description(iter_out)

                pbar.update(1)
                total_c_loss += c_loss_value.data       #c_loss_value.data[0]
                total_accuracy += acc.data              #acc.data[0]

                self.total_train_iter += 1
                if self.total_train_iter % 2000 == 0:
                    self.lr /= 2
                    print("change learning rate", self.lr)

        total_c_loss = total_c_loss / total_train_batches
        total_accuracy = total_accuracy / total_train_batches
        return total_c_loss, total_accuracy

    def run_validation_epoch(self, total_val_batches):
        """
        Runs one validation epoch
        :param total_val_batches: Number of batches to train on
        :return: mean_validation_categorical_crossentropy_loss and mean_validation_accuracy
        """
        total_val_c_loss = 0.
        total_val_accuracy = 0.

        with tqdm.tqdm(total=total_val_batches) as pbar:
            for i in range(total_val_batches):  # validation epoch
                x_support_set, y_support_set, x_target, y_target = \
                    self.data.get_batch(str_type='val', rotate_flag=False)

                x_support_set = Variable(torch.from_numpy(x_support_set), volatile=True).float()
                y_support_set = Variable(torch.from_numpy(y_support_set), volatile=True).long()
                x_target = Variable(torch.from_numpy(x_target), volatile=True).float()
                y_target = Variable(torch.from_numpy(y_target), volatile=True).long()

                # y_support_set: Add extra dimension for the one_hot
                y_support_set = torch.unsqueeze(y_support_set, 2)
                sequence_length = y_support_set.size()[1]
                batch_size = y_support_set.size()[0]
                y_support_set_one_hot = torch.FloatTensor(batch_size, sequence_length,
                                                          self.classes_per_set).zero_()
                y_support_set_one_hot.scatter_(2, y_support_set.data, 1)
                y_support_set_one_hot = Variable(y_support_set_one_hot)

                # Reshape channels
                size = x_support_set.size()
                x_support_set = x_support_set.view(size[0], size[1], size[4], size[2], size[3])
                size = x_target.size()
                x_target = x_target.view(size[0],size[1],size[4],size[2],size[3])
                if self.isCudaAvailable:
                    acc, c_loss_value, _ = self.matchingNet(x_support_set.cuda(), y_support_set_one_hot.cuda(),
                                                         x_target.cuda(), y_target.cuda())
                else:
                    acc, c_loss_value, _ = self.matchingNet(x_support_set, y_support_set_one_hot,
                                                         x_target, y_target)

                #iter_out = "val_loss: {}, val_accuracy: {}".format(c_loss_value.data[0], acc.data[0])
                iter_out = "val_loss: {}, val_accuracy: {}".format(c_loss_value.data, acc.data)
                pbar.set_description(iter_out)
                pbar.update(1)

                total_val_c_loss += c_loss_value.data       #c_loss_value.data[0]
                total_val_accuracy += acc.data              #acc.data[0]

        total_val_c_loss = total_val_c_loss / total_val_batches
        total_val_accuracy = total_val_accuracy / total_val_batches

        return total_val_c_loss, total_val_accuracy

    def run_testing_epoch(self, total_test_batches):
        """
        Runs one testing epoch
        :param total_test_batches: Number of batches to train on
        :param sess: Session object
        :return: mean_testing_categorical_crossentropy_loss and mean_testing_accuracy
        """
        total_test_c_loss = 0.
        total_test_accuracy = 0.
        with tqdm.tqdm(total=total_test_batches) as pbar:
            for i in range(total_test_batches):
                x_support_set, y_support_set, x_target, y_target = \
                    self.data.get_batch(str_type='test', rotate_flag=False)

                x_support_set = Variable(torch.from_numpy(x_support_set), volatile=True).float()
                y_support_set = Variable(torch.from_numpy(y_support_set), volatile=True).long()
                x_target = Variable(torch.from_numpy(x_target), volatile=True).float()
                y_target = Variable(torch.from_numpy(y_target), volatile=True).long()

                # y_support_set: Add extra dimension for the one_hot
                y_support_set = torch.unsqueeze(y_support_set, 2)
                sequence_length = y_support_set.size()[1]
                batch_size = y_support_set.size()[0]
                y_support_set_one_hot = torch.FloatTensor(batch_size, sequence_length,
                                                          self.classes_per_set).zero_()
                y_support_set_one_hot.scatter_(2, y_support_set.data, 1)
                y_support_set_one_hot = Variable(y_support_set_one_hot)

                # Reshape channels
                size = x_support_set.size()
                x_support_set = x_support_set.view(size[0], size[1], size[4], size[2], size[3])
                size = x_target.size()
                x_target = x_target.view(size[0],size[1],size[4],size[2],size[3])
                if self.isCudaAvailable:
                    acc, c_loss_value, _ = self.matchingNet(x_support_set.cuda(), y_support_set_one_hot.cuda(),
                                                         x_target.cuda(), y_target.cuda())
                else:
                    acc, c_loss_value, _ = self.matchingNet(x_support_set, y_support_set_one_hot,
                                                         x_target, y_target)

                #iter_out = "test_loss: {}, test_accuracy: {}".format(c_loss_value.data[0], acc.data[0])
                iter_out = "test_loss: {}, test_accuracy: {}".format(c_loss_value.data, acc.data)
                pbar.set_description(iter_out)
                pbar.update(1)

                total_test_c_loss += c_loss_value.data            #c_loss_value.data[0]
                total_test_accuracy += acc.data                   #acc.data[0]
            total_test_c_loss = total_test_c_loss / total_test_batches
            total_test_accuracy = total_test_accuracy / total_test_batches
        return total_test_c_loss, total_test_accuracy

    def run_time_predictions(self, total_test_batches, is_debug = False):
        """
        Runs one testing epoch
        :param total_test_batches: Number of batches to train on
        :param sess: Session object
        :return: mean_testing_categorical_crossentropy_loss and mean_testing_accuracy
        """
        total_test_c_loss = 0.
        total_test_accuracy = 0.
        with tqdm.tqdm(total=total_test_batches) as pbar:
            for i in range(total_test_batches):
                x_support_set, y_support_set, x_target, y_target = \
                    self.data.get_batch_custom(str_type='x_to_be_predicted', cls=0, rotate_flag=False)

                x_support_set = Variable(torch.from_numpy(x_support_set), volatile=True).float()
                y_support_set = Variable(torch.from_numpy(y_support_set), volatile=True).long()
                x_target = Variable(torch.from_numpy(x_target), volatile=True).float()
                y_target = Variable(torch.from_numpy(y_target), volatile=True).long()

                # y_support_set: Add extra dimension for the one_hot
                y_support_set = torch.unsqueeze(y_support_set, 2)
                sequence_length = y_support_set.size()[1]
                batch_size = y_support_set.size()[0]
                y_support_set_one_hot = torch.FloatTensor(batch_size, sequence_length,
                                                          self.classes_per_set).zero_()
                y_support_set_one_hot.scatter_(2, y_support_set.data, 1)
                y_support_set_one_hot = Variable(y_support_set_one_hot)

                # Reshape channels
                size = x_support_set.size()
                x_support_set = x_support_set.view(size[0], size[1], size[4], size[2], size[3])
                size = x_target.size()
                x_target = x_target.view(size[0],size[1],size[4],size[2],size[3])
                if self.isCudaAvailable:
                    acc, c_loss_value, _ = self.matchingNet(x_support_set.cuda(), y_support_set_one_hot.cuda(),
                                                         x_target.cuda(), y_target.cuda(), is_debug )
                else:
                    acc, c_loss_value, _ = self.matchingNet(x_support_set, y_support_set_one_hot,
                                                         x_target, y_target, is_debug )

                if is_debug == True:
                    #iter_out = "test_loss: {}, test_accuracy: {}".format(c_loss_value.data[0], acc.data[0])
                    iter_out = "test_loss: {}, test_accuracy: {}".format(c_loss_value.data, acc.data)
                    pbar.set_description(iter_out)
                    pbar.update(1)

                total_test_c_loss += c_loss_value.data   #c_loss_value.data[0]
                total_test_accuracy += acc.data          #acc.data[0]
            total_test_c_loss = total_test_c_loss / total_test_batches
            total_test_accuracy = total_test_accuracy / total_test_batches
        return total_test_c_loss, total_test_accuracy
        
    def run_evaluation(self, total_test_batches, is_debug = False):
        """
        Runs one testing epoch
        :param total_test_batches: Number of batches to train on
        :param sess: Session object
        :return: mean_testing_categorical_crossentropy_loss and mean_testing_accuracy
        """
        for i in range(total_test_batches):
            x_support_set, y_support_set, x_target, y_target, target_y_actuals = \
                self.data.get_batch_evaluation(str_type='evaluation', cls=0, rotate_flag=False)
                
            x_support_set = Variable(torch.from_numpy(x_support_set), volatile=True).float()
            y_support_set = Variable(torch.from_numpy(y_support_set), volatile=True).long()
            x_target = Variable(torch.from_numpy(x_target), volatile=True).float()
            y_target = Variable(torch.from_numpy(y_target), volatile=True).long()

            # y_support_set: Add extra dimension for the one_hot
            y_support_set = torch.unsqueeze(y_support_set, 2)
            sequence_length = y_support_set.size()[1]
            batch_size = y_support_set.size()[0]
            y_support_set_one_hot = torch.FloatTensor(batch_size, sequence_length,
                                                      self.classes_per_set).zero_()
            y_support_set_one_hot.scatter_(2, y_support_set.data, 1)
            y_support_set_one_hot = Variable(y_support_set_one_hot)

            # Reshape channels
            size = x_support_set.size()
            x_support_set = x_support_set.view(size[0], size[1], size[4], size[2], size[3])
            size = x_target.size()
            x_target = x_target.view(size[0],size[1],size[4],size[2],size[3])
            if self.isCudaAvailable:
                acc, c_loss_value, pred_indices = self.matchingNet(x_support_set.cuda(), y_support_set_one_hot.cuda(),
                                                     x_target.cuda(), y_target.cuda(), is_debug = False )
            else:
                acc, c_loss_value, pred_indices = self.matchingNet(x_support_set, y_support_set_one_hot,
                                                     x_target, y_target, is_debug = is_debug, is_evaluation_only = False, y_support_set_org = y_support_set, target_y_actuals = target_y_actuals )

            if is_debug == True:
                #iter_out = "test_loss: {}, test_accuracy: {}".format(c_loss_value.data[0], acc.data[0])
                print( "test_loss: {}, test_accuracy: {}".format(c_loss_value.data, acc.data) )
            
            return c_loss_value, acc, x_support_set, y_support_set_one_hot, x_target, y_target, target_y_actuals, pred_indices
            
    def predict(self, total_test_batches, is_debug = False, support_set_images_shape_1_ = -1):
        """
        Runs one testing epoch
        :param total_test_batches: Number of batches to train on
        :param sess: Session object
        :return: mean_testing_categorical_crossentropy_loss and mean_testing_accuracy
        """
        for i in range(total_test_batches):
            x = self.data.get_batch_predict(str_type='evaluation')
                
            x_support_set, y_support_set, x_target, y_target, target_y_actuals = \
                self.data.get_batch_evaluation(str_type='evaluation', cls=0, rotate_flag=False)
            
            if is_debug:
                print( "y_support_set" )
                print( y_support_set )
            
            #x = Variable(torch.from_numpy(x), volatile=True).float()
            
            x_support_set = Variable(torch.from_numpy(x_support_set), volatile=True).float()
            y_support_set = Variable(torch.from_numpy(y_support_set), volatile=True).long()
            x_target = Variable(torch.from_numpy(x_target), volatile=True).float()
            y_target = Variable(torch.from_numpy(y_target), volatile=True).long()

            # y_support_set: Add extra dimension for the one_hot
            y_support_set = torch.unsqueeze(y_support_set, 2)
            sequence_length = y_support_set.size()[1]
            batch_size = y_support_set.size()[0]
            y_support_set_one_hot = torch.FloatTensor(batch_size, sequence_length,
                                                      self.classes_per_set).zero_()
            y_support_set_one_hot.scatter_(2, y_support_set.data, 1)
            y_support_set_one_hot = Variable(y_support_set_one_hot)
            if is_debug:
                print( y_support_set )
                print( y_support_set_one_hot )
            

            # Reshape channels
            size = x.shape
            x = x.reshape(size[0], size[1], size[4], size[2], size[3])
            size = x_support_set.size()
            x_support_set = x_support_set.view(size[0], size[1], size[4], size[2], size[3])
            size = x_target.size()
            x_target = x_target.view(size[0],size[1],size[4],size[2],size[3])
            if self.isCudaAvailable:
                acc, c_loss_value, pred_indices, emcllcls, emcllclsl, emclvlcls, emclvlclsl, open_match_cnt, open_match_mpr = self.matchingNet.predict(x, y_support_set_one_hot.cuda(),
                                                     x_target, y_target.cuda(), is_debug = is_debug, is_evaluation_only = False, y_support_set_org = y_support_set, target_y_actuals = target_y_actuals, support_set_images_shape_1_ = support_set_images_shape_1_ )
            else:
                acc, c_loss_value, pred_indices, emcllcls, emcllclsl, emclvlcls, emclvlclsl, open_match_cnt, open_match_mpr = self.matchingNet.predict(x, y_support_set_one_hot,
                                                     x_target, y_target, is_debug = is_debug, is_evaluation_only = False, y_support_set_org = y_support_set, target_y_actuals = target_y_actuals, support_set_images_shape_1_ = support_set_images_shape_1_ )

            if is_debug == True:
                #iter_out = "test_loss: {}, test_accuracy: {}".format(c_loss_value.data[0], acc.data[0])
                print( "test_loss: {}, test_accuracy: {}".format(c_loss_value.data, acc.data) )
            
            return c_loss_value, acc, x_support_set, y_support_set_one_hot, x_target, y_target, target_y_actuals, pred_indices, emcllcls, emcllclsl, emclvlcls, emclvlclsl, open_match_cnt, open_match_mpr        
        
    def __adjust_learning_rate(self,optimizer):
        """Updates the learning rate given the learning rate decay.
        The routine has been implemented according to the original Lua SGD optimizer
        """
        for group in optimizer.param_groups:
            if 'step' not in group:
                group['step'] = 0
            group['step'] += 1

            group['lr'] = self.lr / (1 + group['step'] * self.lr_decay)

    def __create_optimizer(self,model, new_lr):
        # setup optimizer
        if self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=new_lr,
                                  momentum=0.9, dampening=0.9,
                                  weight_decay=self.wd)
        elif self.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=new_lr,
                                   weight_decay=self.wd)
        else:
            raise Exception('Not supported optimizer: {0}'.format(self.optimizer))
        return optimizer