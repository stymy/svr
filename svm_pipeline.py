import nipype.pipeline.engine as pe
import nipype.interfaces.afni as afni
import nipype.pipeline.utils as util
import nipype.interfaces.io as nio
import nibabel as nb
import numpy as np
import os
from scipy.stats import nanmean, pearsonr

from variables import subjects, sessions, preprocs, workingdir, preproc_directory, ROIs, ROI_file

def get_wf():
    wf = pe.Workflow(name="svr_workflow")
    wf.base_dir = os.path.join(workingdir,"svm_pipeline")
    wf.config['execution']['crashdump_dir'] = wf.base_dir + "/crash_files"

    #INFOSOURCE ITERABLES
    subject_id_infosource = pe.Node(util.IdentityInterface(fields=['subject_id']), name="subject_id_infosource")
    subject_id_infosource.iterables = ('subject_id', subjects)
    
    session_id_infosource = pe.Node(util.IdentityInterface(fields=['session_id']), name="session_id_infosource")
    session_id_infosource.iterables = ('session_id', sessions)
    
    preproc_id_infosource = pe.Node(util.IdentityInterface(fields=['preproc_id']), name="preproc_id_infosource")
    preproc_id_infosource.iterables = ('preproc_id', preprocs)
    
    ROI_id_infosource = pe.Node(util.IdentityInterface(fields=['ROI_id']), name="ROI_id_infosource")
    ROI_id_infosource.iterables = ('ROI_id', ROIs)


    #DATAGRABBER
    #/home/rschadmin/Data/cpac_benchmark/output/pipeline_benchmark_pipeline/0010042_session_1/functional_mni/_scan_rest_1_rest/_csf_threshold_0.96/_gm_threshold_0.7/_wm_threshold_0.96/_compcor_ncomponents_5_selector_pc10.linear0.wm0.global0.motion0.quadratic0.gm0.compcor1.csf0/_bandpass_freqs_0.01.0.1/bandpassed_demeaned_filtered_wtsimt.nii.gz
    datagrabber = pe.Node(nio.DataGrabber(infields=['subject_id','session_id','preproc_id'], outfields=['rest_data']), name='datagrabber')
    datagrabber.inputs.base_directory = '/'
    datagrabber.inputs.template = '*'
    datagrabber.inputs.field_template = dict(rest_data=os.path.join(preproc_directory,'%s_%s/functional_mni/_scan_rest_1_rest/*/*/*/*%s/_bandpass_freqs_0.01.0.1/bandpassed_demeaned_filtered_wtsimt.nii.gz'))#from_paths_file
    datagrabber.inputs.template_args = dict(rest_data=[['subject_id', 'session_id', 'preproc_id']])
    datagrabber.inputs.sort_filelist = True
    
    wf.connect(subject_id_infosource, 'subject_id', datagrabber, 'subject_id')
    wf.connect(session_id_infosource, 'session_id', datagrabber, 'session_id')
    wf.connect(preproc_id_infosource, 'preproc_id', datagrabber, 'preproc_id')
    
    

    #AUTOMASK REST_DATA
    automasker = pe.Node(afni.Automask(), name = 'automasker')
    wf.connect(datagrabber, 'rest_data', automasker, 'in_file')
    
    
    
    #MAKE PREDICTION_MASKS: brain-(ROI + 1voxel perimeter)
    predmasker = pe.Node(afni.Calc(), name = 'predmasker')
    predmasker.inputs.in_file_a = ROI_file
    dilation_str = '-c a+i -d a-i -e a+j -f a-j -g a+k -h a-k' #dilates ROI volume in three directions
    predmasker.inputs.args = '-byte '+dilation_str #byte required for 3dsvm
    predmasker.inputs.outputtype = 'NIFTI'
    
    def get_expr(ROI_id):
        expr = 'and(b,not(amongst(1, equals(a,'+str(ROI_id)+'),c,d,e,f,g,h)))'# exclude ROI and it's outer radius by 1voxel
        return expr
        
    wf.connect(automasker, 'out_file', predmasker, 'in_file_b')
    wf.connect(ROI_id_infosource, ('ROI_id', get_expr), predmasker,'expr')
    
    
    
    #GET TIMESERIES
    TSExtractor = pe.Node(afni.Maskave(), name = 'TSExtractor')
    TSExtractor.inputs.quiet = True
    TSExtractor.inputs.mask = ROI_file
    TSExtractor.inputs.outputtype = 'NIFTI'
    
    def get_mrange(ROI_id):
        mrange = '-mrange '+str(ROI_id)+' '+str(ROI_id)
        return mrange
        
    wf.connect(datagrabber, 'rest_data', TSExtractor, 'in_file')
    wf.connect(ROI_id_infosource, ('ROI_id', get_mrange), TSExtractor, 'args')
    
    
    
    #TRAIN DATA
    trainer = pe.Node(afni.SVMTrain(), name = 'Trainer')
    trainer.inputs.ttype = 'regression'
    trainer.inputs.options = '-c 100 -e 10 -overwrite'
    trainer.inputs.max_iterations = 100
    trainer.inputs.outputtype = 'NIFTI'
    
    wf.connect(datagrabber, 'rest_data', trainer, 'in_file')
    wf.connect(predmasker, 'out_file', trainer, 'mask')
    wf.connect(TSExtractor, 'out_file', trainer, 'trainlabels')
    
    #DATASINK
    ds = pe.Node(nio.DataSink(), name='datasink')
    ds.inputs.base_directory = os.path.join(workingdir,'output')
    
    wf.connect(trainer, 'out_file', ds, 'vectors')
    wf.connect(trainer, 'model', ds, 'models')
    wf.connect(trainer, 'alphas', ds, 'alphas')
    
    return wf
    
if __name__=='__main__':
    wf = get_wf()
    #wf.run(plugin="CondorDAGMan", plugin_args={"template":"universe = vanilla\nnotification = Error\ngetenv = true\nrequest_memory=4000"})
    #wf.run(plugin="MultiProc", plugin_args={"n_procs":16})
    wf.run(plugin="Linear", updatehash=True)    
    
    #TEST DATA

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
        model1 = nb.load('/home/rschadmin/Data/svr_test/w_run'+str(ROI_num)+'.nii').get_data()
        model2 = nb.load('/home/rschadmin/Data/svr_test2/w_run'+str(ROI_num)+'.nii').get_data()
        rep.append(reproducibility(model1,model2))
    np.save('/home/rschadmin/Data/svr_stats/acc1w2',acc1w2)
    np.save('/home/rschadmin/Data/svr_stats/acc2w1',acc2w1)
    np.save('/home/rschadmin/Data/svr_test/rep',rep)
    return acc1w2, acc2w1, rep
