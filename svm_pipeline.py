import nipype.pipeline.engine as pe
import nipype.interfaces.afni as afni
import nibabel as nb
import numpy as np
import os
from scipy.stats import nanmean, pearsonr

from variables import subject_directory

    
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
        #if not os.path.exists(mask):
        dilation_str = '-c a+i -d a-i -e a+j -f a-j -g a+k -h a-k' #dilates ROI volume in three directions
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
        
        #convert models to nifti
        convert = afni.AFNItoNIFTI()
        convert.inputs.in_file = model+'+orig.BRIK'
        convert.inputs.out_file = model+'.nii'
        convert_res = convert.run()

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
        
def conc(x,y,r):
    denominator = x.var(axis=-1)+y.var(axis=-1)+(x.mean(axis=-1)-y.mean(axis=-1))**2
    #mask = np.not_equal(denominator,0)
    #conc = np.choose(mask,(1,(2*rho*x.std(axis=-1)*y.std(axis=-1))/denominator))
    conc = 2.*r*x.std(axis=-1)*y.std(axis=-1)/denominator
    return conc

def rho3D(x,y):
    #truncate x or y to fit same shape
    if x.shape<y.shape:
        y = y[:,:,:,:x.shape[3]]
    else:
        x = x[:,:,:,:y.shape[3]]
    #rho = pearsonr(x.flatten(),y.flatten())[0]
    meanx = x.mean(axis=-1)
    meany = y.mean(axis=-1)
    errx = x-meanx[:,:,:,np.newaxis]
    erry = y-meany[:,:,:,np.newaxis]
    cov = errx*erry
    covariance = cov.sum(axis=3)
    stdx = x.std(axis=-1)
    stdy = y.std(axis=-1)
    rho = covariance/(stdx*stdy)
    return rho
    
# /home/data/Projects/CPAC_Regression_Test/2013-12-05_v-0-3-3/group-correlations_script.py
    
def accuracy(prediction_file, timeseries_file):
    ##Accuracy
    with open(prediction_file) as f:
        prediction = np.fromfile(f, sep=" ")
    with open(timeseries_file) as f:
        timeseries = np.fromfile(f, sep=" ")
    accuracy = conc(prediction, timeseries, pearsonr(prediction,timeseries)[0])
    return accuracy
    
def reproducibility(model1, model2):
    rho = rho3D(model1, model2)
    reproducibility = conc(model1, model2, rho) #nanmean to get one value
    return reproducibility
    
def stats(ROI_data):
    acc1w2 = []
    acc2w1 = []
    rep = []
    for ROI_num in np.unique(ROI_data)[1:]:
        print str(ROI_num)+' of '+str(np.unique(ROI_data).max())
        ##Accuracy 1<-2
        pred_file = '/home/rschadmin/Data/svr_prediction/predict1with2_run'+str(ROI_num)+'.1D'
        TS_file = '/home/rschadmin/Data/svr_test/timeseries'+str(ROI_num)+'.1D'
        acc1w2.append(accuracy(pred_file, TS_file))
        ##Accuracy 1->2
        pred_file = '/home/rschadmin/Data/svr_prediction/predict2with1_run'+str(ROI_num)+'.1D'
        TS_file = '/home/rschadmin/Data/svr_test2/timeseries'+str(ROI_num)+'.1D'
        acc2w1.append(accuracy(pred_file, TS_file))
        ##Reproducibility
        model1 = nb.load('/home/rschadmin/Data/svr_test/model_run'+str(ROI_num)+'.nii').get_data()
        model2 = nb.load('/home/rschadmin/Data/svr_test2/model_run'+str(ROI_num)+'.nii').get_data()
        rep.append(reproducibility(model1,model2))
    np.save('/home/rschadmin/Data/svr_stats/acc1w2',acc1w2)
    np.save('/home/rschadmin/Data/svr_stats/acc2w1',acc2w1)
    np.save('/home/rschadmin/Data/svr_test/rep',rep)
    return acc1w2, acc2w1, rep
