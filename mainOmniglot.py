##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Albert Berenguel
## Computer Vision Center (CVC). Universitat Autonoma de Barcelona
## Email: aberenguel@cvc.uab.es
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from datasets import omniglotNShot
from option import Options
from experiments.OneShotBuilder import OneShotBuilder
import tqdm
from logger import Logger
import os, sys

'''
:param batch_size: Experiment batch_size
:param classes_per_set: Integer indicating the number of classes per set
:param samples_per_class: Integer indicating samples per class
        e.g. For a 20-way, 1-shot learning task, use classes_per_set=20 and samples_per_class=1
             For a 5-way, 10-shot learning task, use classes_per_set=5 and samples_per_class=10
'''

is_debug = True

ENV = int(sys.argv[1]) if len(sys.argv) >= 2 else 0

#use absolute paths
ABS_PATh = os.path.dirname(os.path.abspath(__file__)) + "/"


is_use_sample_data = False

# Experiment Setup
if is_use_sample_data:
    batch_size = 32
    fce = False
    classes_per_set = 5
    samples_per_class = 5
    channels = 1
    # Training setup
    total_epochs = 500
    total_train_batches = 1000
    total_val_batches = 100
    total_test_batches = 250
else:
    batch_size = 8 #32
    fce = False
    classes_per_set = 5
    samples_per_class = 5
    channels = 1
    # Training setup
    total_epochs = 40 #500
    total_train_batches = 50 # 1000
    total_val_batches = 20 # 100
    total_test_batches = 20 # 250

# Parse other options
log_dir = ''
dataroot = ''
if is_use_sample_data:
    args = Options().parse()
    log_dir = args.log_dir
    dataroot = args.dataroot
else:
    dataroot = '/tmp/omniglot'
    log_dir = './logs'

LOG_DIR = log_dir + '/1_run-batchSize_{}-fce_{}-classes_per_set{}-samples_per_class{}-channels{}' \
    .format(batch_size,fce,classes_per_set,samples_per_class,channels)

# create logger
logger = Logger(LOG_DIR)

total_input_files = int(sys.argv[21]) if len(sys.argv) >= 22 else 0

data = omniglotNShot.OmniglotNShotDataset(dataroot=dataroot, batch_size = batch_size,
                                          classes_per_set=classes_per_set,
                                          samples_per_class=samples_per_class, 
                                          is_use_sample_data=is_use_sample_data, input_file=sys.argv[2], input_labels_file=sys.argv[3], 
                                          total_input_files = total_input_files)

obj_oneShotBuilder = OneShotBuilder(data)
obj_oneShotBuilder.build_experiment(batch_size, classes_per_set, samples_per_class, channels, fce)

best_val = 0.
with tqdm.tqdm(total=total_epochs) as pbar_e:
    for e in range(0, total_epochs):
        total_c_loss, total_accuracy = obj_oneShotBuilder.run_training_epoch(total_train_batches=total_train_batches)
        print("Epoch {}: train_loss: {}, train_accuracy: {}".format(e, total_c_loss, total_accuracy))

        total_val_c_loss, total_val_accuracy = obj_oneShotBuilder.run_validation_epoch(
            total_val_batches=total_val_batches)
        print("Epoch {}: val_loss: {}, val_accuracy: {}".format(e, total_val_c_loss, total_val_accuracy))

        logger.log_value('train_loss', total_c_loss)
        logger.log_value('train_acc', total_accuracy)
        logger.log_value('val_loss', total_val_c_loss)
        logger.log_value('val_acc', total_val_accuracy)

        if total_val_accuracy >= best_val:  # if new best val accuracy -> produce test statistics
            best_val = total_val_accuracy
            total_test_c_loss, total_test_accuracy = obj_oneShotBuilder.run_testing_epoch(
                total_test_batches=total_test_batches)
            print("Epoch {}: test_loss: {}, test_accuracy: {}".format(e, total_test_c_loss, total_test_accuracy))
            logger.log_value('test_loss', total_test_c_loss)
            logger.log_value('test_acc', total_test_accuracy)
            
            if True:
                total_test_c_loss, total_test_accuracy = obj_oneShotBuilder.run_time_predictions(
                    total_test_batches=total_test_batches)
                print("Epoch {}: run_time_predictions_loss: {}, run_time_predictions_accuracy: {}".format(e, total_test_c_loss, total_test_accuracy))
                logger.log_value('run_time_predictions_loss', total_test_c_loss)
                logger.log_value('run_time_predictions_acc', total_test_accuracy)
        else:
            total_test_c_loss = -1
            total_test_accuracy = -1

        pbar_e.update(1)
        logger.step()