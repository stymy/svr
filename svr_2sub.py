import nipype.interfaces.afni as afni
import nibabel as nb
import numpy as np
import os
from scipy.stats import nanmean, pearsonr

dataset = 'svr_test5'
testset = 'svr_test6'
filename = "CCB028_20100723_REST.nii.gz"
#inputs
input_data = '/home/rschadmin/Data/'+dataset+'/'+filename
ROIs = '/home/rschadmin/Data/svr_test/rois200_resampled.nii'
total_mask = '/home/rschadmin/Data/'+dataset+'/automask+orig.BRIK'

#find ROI range
ROI_data = nb.load(ROIs).get_data()
def train(input_data, ROI_data, total_mask, ROIs, reverse=False):
    if reverse==True:
        data = testset
        input_data = '/home/rschadmin/Data/'+testset+'/'+filename
        total_mask = '/home/rschadmin/Data/'+testset+'/automask+orig.BRIK'
    else:
        data = dataset
        input_data = '/home/rschadmin/Data/'+dataset+'/'+filename
        total_mask = '/home/rschadmin/Data/'+dataset+'/automask+orig.BRIK'
    #loop through each ROI
    for ROI_num in np.unique(ROI_data)[1:]:
        #set outputs
        timeseries = '/home/rschadmin/Data/'+data+'/timeseries'+str(ROI_num)+'.1D'
        mask = '/home/rschadmin/Data/'+data+'/mask'+str(ROI_num)+'+orig.BRIK'
        model = '/home/rschadmin/Data/'+data+'/model_run'+str(ROI_num)
        w = '/home/rschadmin/Data/'+data+'/w_run'+str(ROI_num)
        alphas = '/home/rschadmin/Data/'+data+'/alphas_run'+str(ROI_num)

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
        training.inputs.options = '-c 1e-6 -e 0.05 -overwrite'
        training.inputs.max_iterations = 1000
        train_res = training.run()
        
        #convert models to nifti
        convert = afni.AFNItoNIFTI()
        convert.inputs.in_file = w+'+orig.BRIK'
        convert.inputs.out_file = w+'.nii'
        convert_res = convert.run()

def test(ROI_data, input_data, reverse=False):
    if reverse == True:
        data1 = testset
        data2 = dataset
    else:
        data1 = dataset
        data2 = testset
    for ROI_num in np.unique(ROI_data)[1:]:
        model = '/home/rschadmin/Data/'+data1+'/model_run'+str(ROI_num)+'+orig.BRIK'
        prediction = '/home/rschadmin/Data/svr_prediction/predict_'+data2+'_with_'+data1+'_run'+str(ROI_num)
        real_model = '/home/rschadmin/Data/'+data2+'/timeseries'+str(ROI_num)+'.1D'
        testing = afni.SVMTest()
        testing.inputs.in_file= '/home/rschadmin/Data/'+data2+'/'+filename
        testing.inputs.model= model
        testing.inputs.testlabels= real_model
        testing.inputs.out_file= prediction
        test_res = testing.run()
        
def conc(x,y,r):
    #denominator = x.var(axis=-1)+y.var(axis=-1)+(x.mean(axis=-1)-y.mean(axis=-1))**2
    denominator = x.var()+y.var()+(x.mean()-y.mean())**2
    #mask = np.not_equal(denominator,0)
    #conc = np.choose(mask,(1,(2*rho*x.std(axis=-1)*y.std(axis=-1))/denominator))
    #conc = 2.*r*x.std(axis=-1)*y.std(axis=-1)/denominator
    conc = 2.*r*x.std()*y.std()/denominator
    return conc

def rho3D(x,y):
    #truncate x or y to fit same shape
    #if x.shape<y.shape:
    #    y = y[:,:,:,:x.shape[3]]
    #else:
    #    x = x[:,:,:,:y.shape[3]]
    rho = pearsonr(x.flatten(),y.flatten())[0]
    #meanx = x.mean(axis=-1)
    #meany = y.mean(axis=-1)
    #errx = x-meanx[:,:,:,np.newaxis]
    #erry = y-meany[:,:,:,np.newaxis]
    #cov = errx*erry
    #covariance = cov.sum(axis=3)
    #stdx = x.std(axis=-1)
    #stdy = y.std(axis=-1)
    #rho = covariance/(stdx*stdy)
    return rho
    
# /home/data/Projects/CPAC_Regression_Test/2013-12-05_v-0-3-3/group-correlations_script.py
    
def accuracy(prediction_file, timeseries_file):
    ##Accuracy
    with open(prediction_file) as f:
        prediction = np.fromfile(f, sep=" ")
    with open(timeseries_file) as f:
        timeseries = np.fromfile(f, sep=" ")
    accuracy = conc(prediction, timeseries, pearsonr(prediction,timeseries)[0])
    print "r=",pearsonr(prediction,timeseries)[0]
    print accuracy
    return accuracy
    
def reproducibility(w1, w2):
    rho = rho3D(w1, w2)
    reproducibility = conc(w1, w2, rho) #nanmean to get one value
    print "r=",rho
    print reproducibility
    return reproducibility
    
def stats(ROI_data):
    acc1w2 = []
    acc2w1 = []
    rep = []
    for ROI_num in np.unique(ROI_data)[1:]:
        print str(ROI_num)+' of '+str(np.unique(ROI_data).max())
        ##Accuracy 1<-2
        pred_file = '/home/rschadmin/Data/svr_prediction/predict_'+testset+'_with_'+dataset+'_run'+str(ROI_num)+'.1D'
        TS_file = '/home/rschadmin/Data/'+testset+'/timeseries'+str(ROI_num)+'.1D'
        acc1w2.append(accuracy(pred_file, TS_file))
        ##Accuracy 1->2
        pred_file = '/home/rschadmin/Data/svr_prediction/predict_'+dataset+'_with_'+testset+'_run'+str(ROI_num)+'.1D'
        TS_file = '/home/rschadmin/Data/'+dataset+'/timeseries'+str(ROI_num)+'.1D'
        acc2w1.append(accuracy(pred_file, TS_file))
        ##Reproducibility
        w1 = nb.load('/home/rschadmin/Data/'+dataset+'/w_run'+str(ROI_num)+'.nii').get_data()
        w2 = nb.load('/home/rschadmin/Data/'+testset+'/w_run'+str(ROI_num)+'.nii').get_data()
        rep.append(reproducibility(w1,w2))
    np.save('/home/rschadmin/Data/svr_stats/acc_'+testset+'_with_'+dataset,acc1w2)
    np.save('/home/rschadmin/Data/svr_stats/acc_'+dataset+'_with_'+testset,acc2w1)
    np.save('/home/rschadmin/Data/svr_test/rep',rep)
    return acc1w2, acc2w1, rep
