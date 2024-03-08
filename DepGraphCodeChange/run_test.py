import subprocess
from tqdm import tqdm
import time
import os, sys
import pickle
import re
import GPUtil
import psutil
import threading
project = sys.argv[1]
trained_model = sys.argv[2]
pp = project
card = [0]



lst = list(range(len(pickle.load(open(project + '.pkl', 'rb')))))
singlenums = {'Time':5, 'Math':2, "Lang":10, "Chart":3, "Mockito":4, "Closure":1, 'Codec':1, 'Compress':1, 'Gson':1, 'Cli':1, 'Jsoup':1, 'Csv':1, 'JacksonCore':1, 'JacksonXml':1, 'Collections':1}
singlenum = singlenums[project]
totalnum = len(card) * singlenum
lr = 1e-2
seed = 0
batch_size = 60
for i in tqdm(range(int(len(lst) / totalnum) + 1)):
    jobs = []
    for j in range(totalnum):
        if totalnum * i + j >= len(lst):
            continue
        cardn =int(j / singlenum)
        p = subprocess.Popen("CUDA_VISIBLE_DEVICES="+str(card[cardn]) + " python3 run_2.py %d %s %f %d %d %s"%(lst[totalnum * i + j], project, lr, seed, batch_size, trained_model), shell=True)
        
        jobs.append(p)
    
        time.sleep(10)
    for p in jobs:
        p.wait()
p = subprocess.Popen("python3 merge_results.py %s %s"%(project, trained_model), shell=True)
p.wait()
# p = subprocess.Popen("python3 top_k.py %s %s" % (project, trained_model), shell=True)
# # Wait for some time or until your condition is met
# p.terminate()  # or p.kill() for a more forceful approach
          