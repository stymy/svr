import pandas as pd
import numpy as np

sublist = \
['20100602',
 '20100608',
 '20100609',
 '20100610',
 '20100611',
 '20100614',
 '20100616',
 '20100618',
 '20100622',
 '20100625',
 '20100709',
 '20100713',
 '20100719',
 '20100719',
 '20100721',
 '20100722',
 '20100723',
 '20100727',
 '20100730',
 '20100817',
 '20100914',
 '20100927',
 '20101001',
 '20101129',
 '20101216']

datalist = []
for subject in sublist:
    for ROI in range(1,5):
        d = dict()
        d['subid'] = 'sub_'+subject
        index = ROI-1
        d['ROI'] = 'r_'+str(ROI)
        one = np.load('/home/rschadmin/Data/CCB/svr_stats/acc_%sScan1_with_%sScan2.npy'%(subject,subject))
        d['accuracy_1 pearson'] = one[index,1]
        d['accuracy_1 concordance'] = one[index,0]
        two = np.load('/home/rschadmin/Data/CCB/svr_stats/acc_%sScan2_with_%sScan1.npy'%(subject,subject))
        d['accuracy_2 pearson'] = two[index,1]
        d['accuracy_2 concordance'] = two[index,0]
        rep = np.load('/home/rschadmin/Data/CCB/svr_stats/rep_%sScan1+2.npy'%(subject))
        d['reproducibility pearson'] = rep[index,1]
        d['reproducibility concordance'] = rep[index,0]
        datalist.append(d)
    
df = pd.DataFrame(datalist)
df.to_csv('/home/rschadmin/Data/CCB/svr_stats/DataFrame.csv')
