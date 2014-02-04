import nipype.interfaces.afni as afni
import nibabel as nb
import numpy as np

dataset = 'svr_test2'

#inputs
input_data = '/home/rschadmin/Data/'+dataset+'/bandpassed_demeaned_filtered_wtsimt.nii.gz'
ROIs = '/home/rschadmin/Data/svr_test/rois200_resampled.nii'
total_mask = '/home/rschadmin/Data/'+dataset+'/automask+orig.BRIK'

#find ROI range
ROI_data = nb.load(ROIs).get_data()
def train(input_data, ROI_data, total_mask):
    #loop through each ROI
    for ROI_num in np.unique(ROI_data):
        #set outputs
        timeseries_data = '/home/rschadmin/Data/'+dataset+'/timeseries'+str(ROI_num)+'.1D'
        mask = '/home/rschadmin/Data/'+dataset+'/mask'+str(ROI_num)+'.nii'
        model = '/home/rschadmin/Data/'+dataset+'/model_run'+str(ROI_num)
        vectors = '/home/rschadmin/Data/'+dataset+'/vectors_run'+str(ROI_num)
        alphas = '/home/rschadmin/Data/'+dataset+'/alphas_run'+str(ROI_num)

        #make mask for not(ROI)
        #3dcalc -a '/home/rschadmin/Data/ROIs/craddock_2011_parcellations/rois200.nii' -expr 'not(equals(a,1))' -byte -prefix mask
        masking = afni.Calc()
        masking.inputs.in_file_a = ROIs
        masking.inputs.in_file_b = total_mask
        ### exclude outer radius 1 voxel (dilate roi)
        masking.inputs.out_file = mask
        masking.inputs.expr = 'and(b,not(equals(a,'+str(ROI_num)+')))'
        masking.inputs.args = '-byte -overwrite'
        maskRun = masking.run()

        #get timeseries (TSE Average ROI)
        #3dmaskave -quiet -mask ~/Data/ROIs/craddock_2011_parcellations/rois200.nii -mrange $i $i resampled_bandpassed_demeaned_filtered+tlrc.BRIK > timeseries$i.nii
        timing = afni.Maskave()
        timing.inputs.in_file = input_data
        timing.inputs.out_file = timeseries_data
        timing.inputs.quiet = True
        timing.inputs.args = '-overwrite -mrange '+str(ROI_num)+' '+str(ROI_num)
        timeRun = timing.run()

        #svm training
        training = afni.SVMTrain()
        training.inputs.in_file= input_data
        training.inputs.out_file= vectors
        training.inputs.trainlabels= timeRun.outputs.out_file
        training.inputs.mask= maskRun.outputs.out_file
        training.inputs.model= model
        training.inputs.alphas= alphas
        training.inputs.ttype = 'regression'
        training.inputs.options = '-c 100 -e 0.001 -overwrite'
        train_res = training.run()

def test(ROI_data, input_data):
    for ROI_num in np.unique(ROI_data):
        model = '/home/rschadmin/Data/svr_test/model_run'+str(ROI_num)+'+orig.BRIK'
        prediction = '/home/rschadmin/Data/svr_prediction/predict2with1_run'+str(ROI_num)
        testing = afni.SVMTest()
        testing.inputs.in_file= input_data
        testing.inputs.model= model
        #testing.inputs.testlabels= real_model
        testing.inputs.out_file= prediction
        test_res = testing.run()
        
def stats():
    print 1
