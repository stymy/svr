import nipype.interfaces.afni as afni
import nibabel as nb
import numpy as np
import os

dataset = 'svr_test'

#inputs
input_data = '/home/rschadmin/Data/'+dataset+'/bandpassed_demeaned_filtered_wtsimt.nii.gz'
ROIs = '/home/rschadmin/Data/svr_test/rois200_resampled.nii'
total_mask = '/home/rschadmin/Data/'+dataset+'/automask+orig.BRIK'

#find ROI range
ROI_data = nb.load(ROIs).get_data()
def train(input_data, ROI_data, total_mask, ROIs):
    #loop through each ROI
    for ROI_num in np.unique(ROI_data)[1:]:
        #set outputs
        timeseries = '/home/rschadmin/Data/'+dataset+'/timeseries'+str(ROI_num)+'.1D'
        mask = '/home/rschadmin/Data/'+dataset+'/mask'+str(ROI_num)+'+orig.BRIK'
        model = '/home/rschadmin/Data/'+dataset+'/model_run'+str(ROI_num)
        w = '/home/rschadmin/Data/'+dataset+'/w_run'+str(ROI_num)
        alphas = '/home/rschadmin/Data/'+dataset+'/alphas_run'+str(ROI_num)

        #make mask for not(ROI)
        #3dcalc -a '/home/rschadmin/Data/ROIs/craddock_2011_parcellations/rois200.nii' -expr 'not(equals(a,1))' -byte -prefix mask
        #if not os.path.exists(mask):
        dilation_str = '-c a+i -d a-i -e a+j -f a-j -g a+k -h a-k' 
        #dilates ROI volume in three directions
        
        masking = afni.Calc()
        masking.inputs.in_file_a = ROIs
        masking.inputs.in_file_b = total_mask
        ### exclude outer radius 1 voxel (dilate roi)
        masking.inputs.out_file = mask
        masking.inputs.expr = 'and(b,not(amongst(1, equals(a,'+str(ROI_num)+'),c,d,e,f,g,h)))' #brain without dilated roi
        masking.inputs.args = '-byte -overwrite '+dilation_str #byte required for 3dsvm
        maskRun = masking.run()

        #get timeseries (TSE Average ROI)
        #3dmaskave -quiet -mask ~/Data/ROIs/craddock_2011_parcellations/rois200.nii -mrange $i $i resampled_bandpassed_demeaned_filtered+tlrc.BRIK > timeseries$i.nii
        timing = afni.Maskave()
        timing.inputs.in_file = input_data
        timing.inputs.out_file = timeseries
        timing.inputs.quiet = True
        timing.inputs.mask = ROIs
        timing.inputs.args = '-overwrite -mrange '+str(ROI_num)+' '+str(ROI_num)
        timeRun = timing.run()

        #svm training
        training = afni.SVMTrain()
        training.inputs.in_file= input_data
        training.inputs.out_file= w
        training.inputs.trainlabels= timeseries
        training.inputs.mask= mask
        training.inputs.model= model
        training.inputs.alphas= alphas
        training.inputs.ttype = 'regression'
        training.inputs.options = '-c 100 -e 0.01 -overwrite'
        training.inputs.max_iterations = 100
        train_res = training.run()

def test(ROI_data, input_data):
    for ROI_num in np.unique(ROI_data)[1:]:
        model = '/home/rschadmin/Data/svr_test2/model_run'+str(ROI_num)+'+orig.BRIK'
        prediction = '/home/rschadmin/Data/svr_prediction/predict1with2_run'+str(ROI_num)
        real_model = '/home/rschadmin/Data/svr_test/timeseries'+str(ROI_num)+'.1D'
        testing = afni.SVMTest()
        testing.inputs.in_file= input_data
        testing.inputs.model= model
        testing.inputs.testlabels= real_model
        testing.inputs.out_file= prediction
        test_res = testing.run()
        
def conc(x,y,rho):
    # equation  
    conc = (2*rho*x.std(axis=-1)*y.std(axis=-1))/(x.var(axis=-1)+y.var(axis=-1)+(x.mean(axis=-1)-y.mean(axis=-1))**2)      

def stats():
    ##Accuracy
    with open('/home/rschadmin/Data/svr_prediction/predict1with2_run1'+str(ROI_num)+'.1D') as f:
        prediction1 = np.fromfile(f, sep=" ")
    with open('/home/rschadmin/Data/svr_test/timeseries'+str(ROI_num)+'.1D') as f:
        timeseries1 = np.fromfile(f, sep=" ")
    accuracy = conc(prediction1, timeseries1, np.corrcoef(x,y)[0,1])
    
    ##Reproducibility
    #load numpy files
    model1_run1 = nb.load('')
    model2_run1 = nb.load('')
    reproducibility = conc(model1_run1, model2_run1, rho)
    
