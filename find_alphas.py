import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt

ROIs = '/home/rschadmin/Data/svr_test/rois200_resampled.nii'
ROI_data = nb.load(ROIs).get_data()

allalphas = []
for ROI_num in np.unique(ROI_data)[1:]:
    alpha_file = 'alphas_run'+str(ROI_num)+'.1D'
    with open(alpha_file) as f:
        alphas = np.fromfile(f,sep=" ")
    allalphas.append(alphas.max())
    
print np.max(allalphas)
allalphas.sort()
plt.hist(allalphas,bins=10)
plt.show()
