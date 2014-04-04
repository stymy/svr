import os
import nibabel as nb
import numpy as np

workingdir = '/home/rschadmin/Data/SVR'
subject_directory = '/home/rschadmin/Data/cpac_benchmark/input/site_1/'
preproc_directory = '/home/rschadmin/Data/cpac_benchmark/output/pipeline_benchmark_pipeline'

ROI_file = '/home/rschadmin/Data/svr_test/rois200_resampled.nii'
ROI_data = nb.load(ROI_file).get_data()
#AFNI 3Dresample'd to resting scan resolution
#3dresample -inset rois200.nii -master bandpassed_demeaned_filtered_wtsimt.nii.gz -prefix rois200_resampled.nii

subjects = os.listdir(subject_directory)
sessions = ['session_1']
preprocs = ['global0.motion0.quadratic0.gm0.compcor1.csf0',
'global1.motion0.quadratic0.gm0.compcor0.csf0']

ROIs = [x for x in np.unique(ROI_data)]

