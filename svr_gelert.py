import nipype.interfaces.afni as afni
import nibabel as nb
import numpy as np
import os
from scipy.stats import nanmean, pearsonr
from shutil import copyfile
import glob
import string
import sys

def train(testset, dataset, filename, ROI_data, ROIs, reverse=False):
    if reverse==True:
        data = testset
        input_data = glob.glob('/home2/data/PreProc/CCB/'+testset+'/*REST2.nii.gz')[0]
    else:
        data = dataset
        input_data = glob.glob('/home2/data/PreProc/CCB/'+dataset+'/*REST1.nii.gz')[0]
    #loop through each ROI
    for ROI_num in np.unique(ROI_data):
        #set outputs
        timeseries = '/home2/data/PreProc/CCB/'+data+'/timeseries'+str(ROI_num)+'.1D'
        mask = '/home2/data/SEEDS/Craddock_2011_parcellations/Craddock_200_rMasks/mask_roi'+str(ROI_num)+'.nii'
        model = '/home2/data/PreProc/CCB/'+data+'/model_run'+str(ROI_num)+'.nii'
        w = '/home2/data/PreProc/CCB/'+data+'/w_run'+str(ROI_num)+'.nii'
        alphas = '/home2/data/PreProc/CCB/'+data+'/alphas_run'+str(ROI_num)

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
        training.inputs.options = '-c 100 -e 0.1 -overwrite'
        training.inputs.max_iterations = 10000
        train_res = training.run()
        
        #convert w's to nifti
        #convert = afni.AFNItoNIFTI()
        #convert.inputs.in_file = w+'+orig.BRIK'
        #convert.inputs.out_file = w+'.nii'
        #convert_res = convert.run()

def test(dataset, testset, filename, ROI_data, reverse=False):
    if reverse == True:
        data1 = testset
        data2 = dataset
    else:
        data1 = dataset
        data2 = testset
    for ROI_num in np.unique(ROI_data):
        model = '/home2/data/PreProc/CCB/'+data1+'/model_run'+str(ROI_num)+'.nii'
        if not os.path.isdir('/home2/data/PreProc/CCB/svr_prediction/'):
            os.mkdir('/home2/data/PreProc/CCB/svr_prediction/')
        prediction = '/home2/data/PreProc/CCB/svr_prediction/predict_'+data2+'_with_'+data1+'_run'+str(ROI_num)
        real_model = '/home2/data/PreProc/CCB/'+data2+'/timeseries'+str(ROI_num)+'.1D'
        testing = afni.SVMTest()
        testing.inputs.in_file= glob.glob('/home2/data/PreProc/CCB/'+data2+'/*.nii.gz')[0]
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
    r = pearsonr(prediction,timeseries)[0]
    accuracy = conc(prediction, timeseries, r)
    print "r=",r
    print accuracy
    return accuracy, r
    
def reproducibility(w1, w2):
    rho = rho3D(w1, w2)
    reproducibility = conc(w1, w2, rho) #nanmean to get one value
    print "r=",rho
    print reproducibility
    return reproducibility, rho
    
def stats(testset, dataset, ROI_data):
    acc1w2 = []
    acc2w1 = []
    rep = []
    for ROI_num in np.unique(ROI_data):
        print str(ROI_num)+' of '+str(np.unique(ROI_data).max())
        ##Accuracy 1<-2

        pred_file = '/home2/data/PreProc/CCB/svr_prediction/predict_'+testset+'_with_'+dataset+'_run'+str(ROI_num)+'.1D'
        TS_file = '/home2/data/PreProc/CCB/'+testset+'/timeseries'+str(ROI_num)+'.1D'
        acc1w2.append(accuracy(pred_file, TS_file))
        ##Accuracy 1->2
        pred_file = '/home2/data/PreProc/CCB/svr_prediction/predict_'+dataset+'_with_'+testset+'_run'+str(ROI_num)+'.1D'
        TS_file = '/home2/data/PreProc/CCB/'+dataset+'/timeseries'+str(ROI_num)+'.1D'
        acc2w1.append(accuracy(pred_file, TS_file))
        ##Reproducibility
        w1 = nb.load('/home2/data/PreProc/CCB/'+dataset+'/w_run'+str(ROI_num)+'.nii').get_data()
        w2 = nb.load('/home2/data/PreProc/CCB/'+testset+'/w_run'+str(ROI_num)+'.nii').get_data()
        rep.append(reproducibility(w1,w2))
    if not os.path.isdir('/home2/data/PreProc/CCB/svr_stats'):
        os.mkdir('/home2/data/PreProc/CCB/svr_stats')
        
    paths = ['/home2/data/PreProc/CCB/svr_stats/acc_'+testset+'_with_'+dataset+'.npy',
             '/home2/data/PreProc/CCB/svr_stats/acc_'+dataset+'_with_'+testset+'.npy',
             '/home2/data/PreProc/CCB/svr_stats/rep_'+dataset+'+2.npy']

    for i, x in enumerate([acc1w2, acc2w1, rep]):
        if ROI_num==1:
            after = x
        else:
            if os.path.exists(paths[i]):
                before = np.load(paths[i])
                after = np.vstack((before,x))
            else: print "NO PREVIOUS FILE"
        np.save(paths[i], after)      ### SAVE: appending new ROI to previous set if not a new run
    
    return acc1w2, acc2w1, rep
    
if __name__ == "__main__":
    datadir = '/home2/data/PreProc/CCB/'
    flist = [f for f in os.listdir(datadir) if f.endswith('1.nii.gz')]
    f = flist[int(sys.argv[1])-1] #'-1' because sge task ids start with 1 indexing
    ROIs = '/home2/data/SEEDS/Craddock_2011_parcellations/craddock200_resampled.nii'
    #find ROI range
    ROI_all = nb.load(ROIs).get_data()
    for ROI in np.unique(ROI_all)[1:]:
        subject_name = f.split('_')[0]
        dataset = subject_name+'Scan1'
        testset = subject_name+'Scan2'
        datasetdir = os.path.join('/home2/data/PreProc/CCB',dataset)
        testsetdir = os.path.join('/home2/data/PreProc/CCB',testset)
        f_match = glob.glob(os.path.join(datadir+'*'+subject_name+'*_REST2.nii.gz'))[0]
        
        if not os.path.isdir(datasetdir):
            os.mkdir(datasetdir)
            copyfile(os.path.join(datadir,f), os.path.join(datasetdir,f))
            copyfile(os.path.join(datadir,f+"+automask.nii"), os.path.join(datasetdir,f+"+automask.nii"))
        if not os.path.isdir(testsetdir):
            os.mkdir(testsetdir)
            copyfile(f_match, os.path.join(testsetdir,os.path.basename(f_match)))
            copyfile(f_match+"+automask.nii", os.path.join(testsetdir,os.path.basename(f_match)+"+automask.nii"))
        train(testset, dataset, f, ROI, ROIs)
        train(testset, dataset, f, ROI, ROIs, reverse=True)
        test(testset, dataset, f, ROI)
        test(testset, dataset, f,ROI, reverse=True)
        stats(testset, dataset, ROI)
