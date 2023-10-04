import subprocess
from tqdm import tqdm
import time
import os, sys
import re
import pickle
project = sys.argv[1]
pp = sys.argv[1]

# Check if the file exists, if yes, delete it
if os.path.exists(f'{pp}_timing_data.txt'):
    os.remove(f'{pp}_timing_data.txt')

card = [0]
lst = list(range(len(pickle.load(open(project + '.pkl', 'rb')))))
singlenums = {'Time':5, 'Math':2, "Lang":1, "Chart":3, "Mockito":4, "Closure":1, "Codec":1, 'Compress':1, 'Gson':1, 'Cli':1, 'Jsoup':1, 'Csv':1, 'JacksonCore':1, 'JacksonXml':1, 'Collections':1}
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
        p = subprocess.Popen("CUDA_VISIBLE_DEVICES="+str(card[cardn]) + " python3 run.py %d %s %f %d %d"%(lst[totalnum * i + j], project, lr, seed, batch_size), shell=True)
        jobs.append(p)
        time.sleep(10)
    for p in jobs:
        p.wait()



p = subprocess.Popen("python3 sum.py %s %d %f %d"%(project, seed, lr, batch_size), shell=True)
p.wait()
subprocess.Popen("python3 watch.py %s %d %f %d"%(project, seed, lr, batch_size),shell=True)       
p.wait()
# After all subprocesses are complete
training_times = []
testing_times = []

# Read the timing data from the file
with open(f'{pp}_timing_data.txt', 'r') as f:
    lines = f.readlines()

    for line in lines:
        match = re.search(r"TIMING_INFO: Training Time: (\d+.\d+), Testing Time: (\d+.\d+)", line)
        if match:
            training_times.append(float(match.group(1)))
            testing_times.append(float(match.group(2)))

# Calculate the total training and testing time
total_training_time = sum(training_times)
total_testing_time = sum(testing_times)

print(f"The overall training time is {total_training_time} seconds.")
print(f"The overall testing time is {total_testing_time} seconds.")