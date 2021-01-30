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
import numpy as np

'''
:param batch_size: Experiment batch_size
:param classes_per_set: Integer indicating the number of classes per set
:param samples_per_class: Integer indicating samples per class
        e.g. For a 20-way, 1-shot learning task, use classes_per_set=20 and samples_per_class=1
             For a 5-way, 10-shot learning task, use classes_per_set=5 and samples_per_class=10
'''

is_debug = True

ENV = int(sys.argv[1]) 

#use absolute paths
ABS_PATh = os.path.dirname(os.path.abspath(__file__)) + "/"


is_use_sample_data = False
is_run_validation_batch = False 
is_run_time_predictions = True if int(sys.argv[34]) == 1 else False 
is_evaluation_res_in_obj = True if int(sys.argv[32]) == 1 else False
is_visualize_data = False
save_interval = int(sys.argv[39])

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
    batch_size = 11 #7     #8 #32
    fce = False
    classes_per_set = 11    #7     #2    #5     #20    #5
    samples_per_class = 1   #5
    channels = 1
    # Training setup
    total_epochs = int(sys.argv[28])    #40 #500
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

model_path = sys.argv[18] if len(sys.argv) >= 19 else 0
outfile_path_prob = sys.argv[19] if len(sys.argv) >= 20 else 0
total_input_files = int(sys.argv[21]) if len(sys.argv) >= 22 else 0

resume_from_epoch = int(sys.argv[49]) if len(sys.argv) >= 50 else -1
is_evaluation_only = False
if os.path.exists(model_path) and resume_from_epoch == -1:
    is_evaluation_only = True
    is_debug = False
elif not resume_from_epoch == -1:
    if not os.path.exists(model_path.replace('EPOCH',str(resume_from_epoch))):
        raise Exception("To resume training the model path is required.")
        
is_do_train_logging = True
if len(sys.argv) >= 51 and int(sys.argv[50]) == 0: 
    is_do_train_logging = False
    
is_do_train_logging_conditionally = True
if len(sys.argv) >= 52 and int(sys.argv[51]) == 0: 
    is_do_train_logging_conditionally = False
    
test_batch_records = 20
if len(sys.argv) >= 53: 
    test_batch_records = int(sys.argv[52])

is_load_test_record = False if int(sys.argv[41]) == 0 else True
test_record_class = int(sys.argv[42])
test_record_index = int(sys.argv[43])
test_record_index_end = int(sys.argv[46])

if is_evaluation_only == False or not is_load_test_record or not test_record_class == -1:
    data = omniglotNShot.OmniglotNShotDataset(dataroot=dataroot, batch_size = batch_size,
                                              classes_per_set=classes_per_set,
                                              samples_per_class=samples_per_class, 
                                              is_use_sample_data=is_use_sample_data, input_file=sys.argv[2], input_labels_file=sys.argv[3], 
                                              total_input_files = total_input_files, is_evaluation_only = is_evaluation_only, 
                                              evaluation_input_file = sys.argv[8], evaluation_labels_file = sys.argv[14], 
                                              evaluate_classes = int(sys.argv[25]), is_eval_with_train_data = int(sys.argv[26]), 
                                              negative_test_offset = int(sys.argv[27]), is_apply_pca_first = int(sys.argv[29]), 
                                              cache_samples_for_evaluation = int(sys.argv[30]), 
                                              is_run_time_predictions = is_run_time_predictions, pca_components = int(sys.argv[31]), 
                                              is_evaluation_res_in_obj = is_evaluation_res_in_obj, total_base_classes =int(sys.argv[33]), 
                                              is_visualize_data = is_visualize_data, is_run_validation_batch = is_run_validation_batch, 
                                              is_compare = False if int(sys.argv[40]) == 0 else True, 
                                              is_load_test_record = is_load_test_record, 
                                              test_record_class = test_record_class, test_record_index = test_record_index, 
                                              is_debug = is_debug, is_switch_dim = False if int(sys.argv[48]) == 0 else True, test_batch_records=test_batch_records)

    obj_oneShotBuilder = OneShotBuilder(data,model_path=model_path if resume_from_epoch == -1 else model_path.replace('EPOCH',str(resume_from_epoch)),model_save_path=model_path)
    obj_oneShotBuilder.build_experiment(batch_size, classes_per_set, samples_per_class, channels, fce, 
                                        image_size = int(sys.argv[35]), layer_size = int(sys.argv[36]), 
                                        is_use_lstm_layer=False if int(sys.argv[37]) == 0 else True, 
                                        vector_dim = int(sys.argv[38]), num_layers=int(sys.argv[44]), dropout=float(sys.argv[45]),
                                        is_use_second_lstm =False if int(sys.argv[47]) == 0 else True, is_do_train_logging=is_do_train_logging, is_do_train_logging_conditionally=is_do_train_logging_conditionally )

if is_evaluation_only == False:
    if not 'EPOCH' in model_path:
        print("Please define model path file name properly with a -EPOCH- key inside so that model can be saved for each epoch")
        
    best_val = 0.
    with tqdm.tqdm(total=total_epochs) as pbar_e:
        for e in range(0 if resume_from_epoch == -1 else resume_from_epoch+1, total_epochs):
            total_c_loss, total_accuracy = obj_oneShotBuilder.run_training_epoch(total_train_batches=total_train_batches, epoch=e)
            print("Epoch {}: train_loss: {}, train_accuracy: {}".format(e, total_c_loss, total_accuracy))

            logger.log_value('train_loss', total_c_loss)
            logger.log_value('train_acc', total_accuracy)

            if is_run_validation_batch:
                total_val_c_loss, total_val_accuracy = obj_oneShotBuilder.run_validation_epoch(
                    total_val_batches=total_val_batches)
                print("Epoch {}: val_loss: {}, val_accuracy: {}".format(e, total_val_c_loss, total_val_accuracy))

                logger.log_value('val_loss', total_val_c_loss)
                logger.log_value('val_acc', total_val_accuracy)

            if False and total_val_accuracy >= best_val:  # if new best val accuracy -> produce test statistics
                best_val = total_val_accuracy
                total_test_c_loss, total_test_accuracy = obj_oneShotBuilder.run_testing_epoch(
                    total_test_batches=total_test_batches)
                print("Epoch {}: test_loss: {}, test_accuracy: {}".format(e, total_test_c_loss, total_test_accuracy))
                logger.log_value('test_loss', total_test_c_loss)
                logger.log_value('test_acc', total_test_accuracy)
                
            else:
                total_test_c_loss = -1
                total_test_accuracy = -1

            if data.is_run_time_predictions:
                total_test_c_loss, total_test_accuracy = obj_oneShotBuilder.run_time_predictions(
                    total_test_batches=total_test_batches, is_debug = is_debug )
                print("Epoch {}: run_time_predictions_loss: {}, run_time_predictions_accuracy: {}".format(e, total_test_c_loss, total_test_accuracy))
                logger.log_value('run_time_predictions_loss', total_test_c_loss)
                logger.log_value('run_time_predictions_acc', total_test_accuracy)
                
            pbar_e.update(1)
            logger.step()
            
            #save model 
            if save_interval == -1 or np.mod( e, save_interval ) == 0:
                obj_oneShotBuilder.save_model(e)
else: 
    is_do_plain_predict = True
    
    print( "is_do_plain_predict ", is_do_plain_predict, " is_evaluation_res_in_obj ", is_evaluation_res_in_obj, " is_load_test_record ", is_load_test_record )
    
    if not is_evaluation_res_in_obj:
        results = []
        resdict = {}
        sloop = int( int(sys.argv[30])/10 )
        for c in range(0, sloop):  #9):
            tot_acc = 0.0
            cnt = 0
            tot_matches = 0
            matched_cnt = 0
            evaluation_cnt = 0
            evaluation_matched_cnt = 0

            for i in range(10):
                if is_debug == True:
                    print( "evaluation i " + str(i) )
                #TODO what if we set support set to empty since its evaluation
                #total_test_c_loss, total_test_accuracy = obj_oneShotBuilder.run_evaluation(total_test_batches=1)
                c_loss_value, acc, x_support_set, y_support_set_one_hot, x_target, y_target, target_y_actuals, pred_indices = obj_oneShotBuilder.run_evaluation(total_test_batches=1, is_debug = is_debug)
                
                tot_acc = tot_acc + acc
                cnt = cnt + 1
                evaluation_cnt = evaluation_cnt + ( (target_y_actuals < 0).sum() )
                
                lenta = len(target_y_actuals[0])
                for j in range(0, lenta):
                    lentai = len(target_y_actuals)
                    for k in range(0, lentai):
                        tot_matches = tot_matches + 1
                        if pred_indices[j][k] == y_target[k][j]:
                            matched_cnt = matched_cnt + 1
                            if target_y_actuals[k][j] < 0:
                                evaluation_matched_cnt = evaluation_matched_cnt + 1
                
                if is_debug == True:
                    #print("predictions loss: {}, predictions_accuracy: {}".format(total_test_c_loss, total_test_accuracy))
                    print(c_loss_value, acc)    #, y_support_set_one_hot, y_target)
                    #print(target_y_actuals)
                    #logger.log_value('run_time_predictions_loss', total_test_c_loss)
                    #logger.log_value('run_time_predictions_acc', total_test_accuracy)
            
            if is_debug == True:        
                print( "class " + str(c) )
                print( "tot_matches " + str( tot_matches ) )
                print( "matched_cnt " + str( matched_cnt ) )
                print( "evaluation_cnt " + str( evaluation_cnt ) )
                print( "evaluation_matched_cnt " + str( evaluation_matched_cnt ) )
                print( "avg acc " + str( (tot_acc / cnt) ) )

            if len(data.shuffle_classes) > 0:
                resdict[data.shuffle_classes[c]] = str( (evaluation_matched_cnt / evaluation_cnt) )
            results.append( str( (evaluation_matched_cnt / evaluation_cnt) ) )
        
        print(resdict)
        print(results)
        
        #save result
        import json
        if not outfile_path_prob == None:
            with open( outfile_path_prob, 'w') as outfile:
                json.dump(results, outfile)                      
    else:
        results = {}
        #resdict = {}
        tot_acc = 0.0
        cnt = 0
        tot_matches = 0
        matched_cnt = 0
        if is_do_plain_predict:
            if True or is_load_test_record:
                is_dynamic_batch = False
            
                test_results = {}
                test_results["mcnt"] = {}
                test_results["micnt"] = {}
                test_results["ci_start_ind"] = 0
                test_results["ri_start_ind"] = 0
                
                if is_load_test_record: 
                    if test_record_class == -1:
                        is_dynamic_batch = True

                        arangec = np.arange( int(sys.argv[33]) )
                        aranger = np.arange( test_record_index, test_record_index_end )   #till available
                    else:
                        arangec = np.array( [ test_record_class ] )
                        aranger = np.array( [ test_record_index ] )
                else: 
                        arangec = np.array( [ 0 ] )
                        aranger = np.array( [ 0 ] )
                
                if is_load_test_record: 
                    if is_dynamic_batch:
                        base_classes_file = sys.argv[2]+"_base_classes.json"
                        #hiren added on 08-06-2020, for when large files are not saved intentionally
                        if not os.path.exists(base_classes_file):
                            bcfdata = omniglotNShot.OmniglotNShotDataset(dataroot=dataroot, batch_size = batch_size,
                                                                      classes_per_set=classes_per_set,
                                                                      samples_per_class=samples_per_class, 
                                                                      is_use_sample_data=is_use_sample_data, input_file=sys.argv[2], input_labels_file=sys.argv[3], 
                                                                      total_input_files = total_input_files, is_evaluation_only = is_evaluation_only, 
                                                                      evaluation_input_file = sys.argv[8], evaluation_labels_file = sys.argv[14], 
                                                                      evaluate_classes = int(sys.argv[25]), is_eval_with_train_data = int(sys.argv[26]), 
                                                                      negative_test_offset = int(sys.argv[27]), is_apply_pca_first = int(sys.argv[29]), 
                                                                      cache_samples_for_evaluation = int(sys.argv[30]), 
                                                                      is_run_time_predictions = is_run_time_predictions, pca_components = int(sys.argv[31]), 
                                                                      is_evaluation_res_in_obj = is_evaluation_res_in_obj, total_base_classes =int(sys.argv[33]), 
                                                                      is_visualize_data = is_visualize_data, is_run_validation_batch = is_run_validation_batch, 
                                                                      is_compare = False if int(sys.argv[40]) == 0 else True, 
                                                                      is_load_test_record = is_load_test_record, 
                                                                      test_record_class = 0, test_record_index = 0, 
                                                                      is_debug = is_debug, is_load_file_data_only = True).get_data_x()
                        
                            #bcfdata = data.x    #omniglotNShot.base_classes_file_data( base_classes_file )
                        else:
                            bcfdata = omniglotNShot.base_classes_file_data( base_classes_file )
                        
                    import os, sys
                    import json
                    test_file_path = './__data/test_results_'+os.path.basename(model_path)+'.json'
                    if os.path.exists( test_file_path ):
                        test_results = json.load( open( test_file_path ) ) 
                        
                        #test_results["mcnt"][str(arangec[test_results["ci_start_ind"]])] = 0
                        #test_results["micnt"][str(arangec[test_results["ci_start_ind"]])] = 0
                        #test_results["ri_start_ind"] = 0
                        
                    
                for ci in range(test_results["ci_start_ind"], arangec.shape[0]):
                    #if test_results["ri_start_ind"] == aranger.shape[0]:
                    #    test_results["ri_start_ind"] = 0
                    
                    if is_load_test_record:                     
                        if is_dynamic_batch:
                            for bcfi in range(0, bcfdata.shape[1]):
                                if np.all( bcfdata[arangec[ci]][bcfi] == 0.0 ):
                                    aranger = np.arange( bcfi-test_batch_records, bcfi-test_batch_records+test_record_index_end )
                                    print( "aranger ", aranger )
                                    break
                    
                    for ri in range(test_results["ri_start_ind"], aranger.shape[0]):
                        try:
                            is_debug = True
                            
                            if is_load_test_record: 
                                data = omniglotNShot.OmniglotNShotDataset(dataroot=dataroot, batch_size = batch_size,
                                                                          classes_per_set=classes_per_set,
                                                                          samples_per_class=samples_per_class, 
                                                                          is_use_sample_data=is_use_sample_data, input_file=sys.argv[2], input_labels_file=sys.argv[3], 
                                                                          total_input_files = total_input_files, is_evaluation_only = is_evaluation_only, 
                                                                          evaluation_input_file = sys.argv[8], evaluation_labels_file = sys.argv[14], 
                                                                          evaluate_classes = int(sys.argv[25]), is_eval_with_train_data = int(sys.argv[26]), 
                                                                          negative_test_offset = int(sys.argv[27]), is_apply_pca_first = int(sys.argv[29]), 
                                                                          cache_samples_for_evaluation = int(sys.argv[30]), 
                                                                          is_run_time_predictions = is_run_time_predictions, pca_components = int(sys.argv[31]), 
                                                                          is_evaluation_res_in_obj = is_evaluation_res_in_obj, total_base_classes =int(sys.argv[33]), 
                                                                          is_visualize_data = is_visualize_data, is_run_validation_batch = is_run_validation_batch, 
                                                                          is_compare = False if int(sys.argv[40]) == 0 else True, 
                                                                          is_load_test_record = is_load_test_record, 
                                                                          test_record_class = arangec[ci], test_record_index = aranger[ri], 
                                                                          is_debug = is_debug)

                                obj_oneShotBuilder = OneShotBuilder(data,model_path=model_path)
                                obj_oneShotBuilder.build_experiment(batch_size, classes_per_set, samples_per_class, channels, fce, 
                                                                    image_size = int(sys.argv[35]), layer_size = int(sys.argv[36]), 
                                                                    is_use_lstm_layer=False if int(sys.argv[37]) == 0 else True, 
                                                                    vector_dim = int(sys.argv[38]), num_layers=int(sys.argv[44]), dropout=float(sys.argv[45]) )

                            support_set_images_shape_1_Lcl = 1
                            if is_load_test_record: 
                                support_set_images_shape_1_Lcl = data.tvt_records_fall_short_clss[arangec[ci]]
                                                                    
                            c_loss_value, acc, x_support_set, y_support_set_one_hot, x_target, y_target, target_y_actuals, pred_indices, emcllcls, emcllclsl, emclvlcls, emclvlclsl, open_match_cnt, open_match_mpr = obj_oneShotBuilder.predict(total_test_batches=1, is_debug = is_debug, support_set_images_shape_1_ = support_set_images_shape_1_Lcl)
                            
                            #
                            for li in range(0, len(emclvlcls)):
                                y_actual = emclvlcls[li]
                                if not y_actual in results:
                                    results[y_actual] = {}
                                    results[y_actual]["ec"] = 1
                                    results[y_actual]["emc"] = 0
                                    results[y_actual]["pr"] = 0.0
                                else:
                                    results[y_actual]["ec"] = results[y_actual]["ec"] + 1
                                    
                                #results[y_actual]["pr"] = 0 + results[y_actual]["pr"] + ( (1.0 - emclvlclsl[li].item()) )
                                results[y_actual]["pr"] = 0 + results[y_actual]["pr"] + ( (emclvlclsl[li].item()) )
                            
                            for li in range(0, len(emcllcls)):
                                y_actual = emcllcls[li]
                                if not y_actual in results:
                                    results[y_actual] = {}
                                    results[y_actual]["ec"] = 1
                                    results[y_actual]["emc"] = 0
                                    results[y_actual]["pr"] = 0.0
                                else:
                                    results[y_actual]["ec"] = results[y_actual]["ec"] + 1
                                    
                                #results[y_actual]["pr"] = 0 + results[y_actual]["pr"] + (1.0 - emcllclsl[li].item())
                                results[y_actual]["pr"] = 0 + results[y_actual]["pr"] + (emcllclsl[li].item())

                            print( "class ", arangec[ci], " record ", aranger[ri], " open_match_cnt ", open_match_cnt, " open_match_mpr ", open_match_mpr )
                                
                            print(results)
                            
                            arangecLcl = arangec 
                            if not is_load_test_record: 
                                arangecLcl = np.arange( int(sys.argv[33]) )
                            
                            #
                            if str(arangecLcl[ci])+"_"+str(arangecLcl[ci]) not in test_results["mcnt"]:
                                #test_results["mcnt"][str(arangec[ci])] = 0
                                #test_results["micnt"][str(arangec[ci])] = 0
                                for ciitmp in range(0, arangecLcl.shape[0]):
                                    #if not ciitmp == ci:
                                    test_results["mcnt"][str(arangecLcl[ciitmp])+"_"+str(arangecLcl[ci])] = 0
                                    test_results["micnt"][str(arangecLcl[ciitmp])+"_"+str(arangecLcl[ci])] = 0
                                
                            #test_results["mcnt"][str(arangec[ci])] += open_match_cnt[arangec[ci]]
                            #test_results["micnt"][str(arangec[ci])] += 1 if open_match_cnt[arangec[ci]] > 0 else 0
                            for ciitmp in range(0, arangecLcl.shape[0]):
                                #if not ciitmp == ci:
                                test_results["mcnt"][str(arangecLcl[ciitmp])+"_"+str(arangecLcl[ci])] += open_match_cnt[arangecLcl[ciitmp]]
                                test_results["micnt"][str(arangecLcl[ciitmp])+"_"+str(arangecLcl[ci])] += 1 if open_match_cnt[arangecLcl[ciitmp]] > 0 else 0
                                
                                results[arangecLcl[ciitmp]] = {}
                                results[arangecLcl[ciitmp]]["ec"] = 2   #set 2 based on older logic in else block below, so not sure if it is still good, but it is not used mostly
                                results[arangecLcl[ciitmp]]["emc"] = 0
                                results[arangecLcl[ciitmp]]["pr"] = ( open_match_cnt[arangecLcl[ciitmp]] / 100 )
                            
                            if not is_load_test_record: 
                                print(results)
                            
                            if is_load_test_record: 
                                #write to file
                                test_results["ci_start_ind"] = ci
                                test_results["ri_start_ind"] = ri
                                print( "test_results ", test_results )
                                with open(test_file_path, 'w') as f:
                                    json.dump(test_results, f)
                            
                        except Exception as e:
                            import traceback
                            print(traceback.format_exc())

                    if is_load_test_record:                             
                        test_results["ri_start_ind"] = 0

                if is_load_test_record:                         
                    for ci in range(0, arangec.shape[0]):
                        ##test_results["mcnt"][str(arangec[ci])] = round( ( test_results["mcnt"][str(arangec[ci])] / (aranger.shape[0]*data.tvt_records) ) * 100, 2 )
                        #test_results["mcnt"][str(arangec[ci])] = round( ( test_results["mcnt"][str(arangec[ci])] / (aranger.shape[0]*data.tvt_records_fall_short_clss[arangec[ci]]) ) * 100, 2 )
                        #test_results["micnt"][str(arangec[ci])] = round( ( test_results["micnt"][str(arangec[ci])] / (aranger.shape[0]) ) * 100, 2 )
                        for ciitmp in range(0, arangec.shape[0]):
                            ##if not ciitmp == ci:
                            #test_results["mcnt"][str(arangec[ciitmp])+"_"+str(arangec[ci])] += 0
                            #test_results["micnt"][str(arangec[ciitmp])+"_"+str(arangec[ci])] += 0
                            test_results["mcnt"][str(arangec[ciitmp])+"_"+str(arangec[ci])] = round( ( test_results["mcnt"][str(arangec[ciitmp])+"_"+str(arangec[ci])] / (aranger.shape[0]*data.tvt_records_fall_short_clss[arangec[ci]]) ) * 100, 2 )
                            test_results["micnt"][str(arangec[ciitmp])+"_"+str(arangec[ci])] = round( ( test_results["micnt"][str(arangec[ciitmp])+"_"+str(arangec[ci])] / (aranger.shape[0]) ) * 100, 2 )
                        
                    print(test_results)
                    if os.path.exists( test_file_path ):
                        os.remove( test_file_path ) 
                            
            else:
                
                #keep debug off in predict mode 
                is_debug = False
            
                c_loss_value, acc, x_support_set, y_support_set_one_hot, x_target, y_target, target_y_actuals, pred_indices, emcllcls, emcllclsl, emclvlcls, emclvlclsl, open_match_cnt, open_match_mpr = obj_oneShotBuilder.predict(total_test_batches=1, is_debug = is_debug, support_set_images_shape_1_ = 1)
                
                #
                for li in range(0, len(emclvlcls)):
                    y_actual = emclvlcls[li]
                    if not y_actual in results:
                        results[y_actual] = {}
                        results[y_actual]["ec"] = 1
                        results[y_actual]["emc"] = 0
                        results[y_actual]["pr"] = 0.0
                    else:
                        results[y_actual]["ec"] = results[y_actual]["ec"] + 1
                        
                    #results[y_actual]["pr"] = 0 + results[y_actual]["pr"] + ( (1.0 - emclvlclsl[li].item()) )
                    results[y_actual]["pr"] = 0 + results[y_actual]["pr"] + ( (emclvlclsl[li].item()) )
                
                for li in range(0, len(emcllcls)):
                    y_actual = emcllcls[li]
                    if not y_actual in results:
                        results[y_actual] = {}
                        results[y_actual]["ec"] = 1
                        results[y_actual]["emc"] = 0
                        results[y_actual]["pr"] = 0.0
                    else:
                        results[y_actual]["ec"] = results[y_actual]["ec"] + 1
                        
                    #results[y_actual]["pr"] = 0 + results[y_actual]["pr"] + (1.0 - emcllclsl[li].item())
                    results[y_actual]["pr"] = 0 + results[y_actual]["pr"] + (emcllclsl[li].item())

                print( "open_match_cnt ", open_match_cnt, " open_match_mpr ", open_match_mpr )
                    
                for key in open_match_cnt.keys():
                    if not key in results:
                        results[key] = {}
                
                    results[key]["ec"] = 2
                    results[key]["pr"] = str( ( 0 + open_match_cnt[key] / 25 ) + ( open_match_mpr[key].item() if open_match_mpr[key] > 0 else 0.0 ) )
                    
                print(results)
        
        else:
            sloop = int( int(sys.argv[30])/10 )
            for c in range(0, sloop):  #9):
                #evaluation_cnt = 0
                #evaluation_matched_cnt = 0

                for i in range(10):
                    if is_debug == True:
                        print( "evaluation i " + str(i) )
                    #TODO what if we set support set to empty since its evaluation
                    #total_test_c_loss, total_test_accuracy = obj_oneShotBuilder.run_evaluation(total_test_batches=1)
                    c_loss_value, acc, x_support_set, y_support_set_one_hot, x_target, y_target, target_y_actuals, pred_indices = obj_oneShotBuilder.run_evaluation(total_test_batches=1, is_debug = is_debug)
                    
                    tot_acc = tot_acc + acc
                    cnt = cnt + 1
                    #evaluation_cnt = evaluation_cnt + ( (target_y_actuals < 0).sum() )
                    
                    lenta = len(target_y_actuals[0])
                    for j in range(0, lenta):
                        lentai = len(target_y_actuals)
                        for k in range(0, lentai):
                            tot_matches = tot_matches + 1
                            
                            #
                            y_actual = -1
                            if target_y_actuals[k][j] < 0:
                                y_actual = ( target_y_actuals[k][j] * -1 ) - 1
                                if not y_actual in results:
                                    results[y_actual] = {}
                                    results[y_actual]["ec"] = 1
                                    results[y_actual]["emc"] = 0
                                    results[y_actual]["pr"] = 0
                                else:
                                    results[y_actual]["ec"] = results[y_actual]["ec"] + 1
                                
                            if pred_indices[j][k] == y_target[k][j]:
                                matched_cnt = matched_cnt + 1
                                if target_y_actuals[k][j] < 0:
                                    #evaluation_matched_cnt = evaluation_matched_cnt + 1
                                    results[y_actual]["emc"] = results[y_actual]["emc"] + 1
                    
                    if is_debug == True:
                        #print("predictions loss: {}, predictions_accuracy: {}".format(total_test_c_loss, total_test_accuracy))
                        print(c_loss_value, acc)    #, y_support_set_one_hot, y_target)
                        #print(target_y_actuals)
                        #logger.log_value('run_time_predictions_loss', total_test_c_loss)
                        #logger.log_value('run_time_predictions_acc', total_test_accuracy)
                
                if is_debug == True:        
                    print( "class " + str(c) )
                    print( "tot_matches " + str( tot_matches ) )
                    print( "matched_cnt " + str( matched_cnt ) )
                    #print( "evaluation_cnt " + str( evaluation_cnt ) )
                    #print( "evaluation_matched_cnt " + str( evaluation_matched_cnt ) )
                    print( "avg acc " + str( (tot_acc / cnt) ) )

                #if len(data.shuffle_classes) > 0:
                #    resdict[data.shuffle_classes[c]] = str( (evaluation_matched_cnt / evaluation_cnt) )
                #results.append( str( (evaluation_matched_cnt / evaluation_cnt) ) )
            
            #print(resdict)
            for key in results.keys():
                if True or is_debug == True:        
                    if (results[key]["emc"] / results[key]["ec"]) >= 0.5:
                        print( "ckey " + str(key) + " pr " + str( results[key]["emc"] / results[key]["ec"] ) )
                    
                results[key]["pr"] = str( results[key]["emc"] / results[key]["ec"] )
            print(results)
