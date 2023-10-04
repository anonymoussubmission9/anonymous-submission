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
pp = project
card = [0]
######
def get_gpu_memory():
    GPUs = GPUtil.getGPUs()
    gpu = GPUs[0]  # assuming you want to monitor the first GPU
    return gpu.memoryUsed


def print_gpu_memory_usage():
    while True:
        GPUs = GPUtil.getGPUs()
        for gpu in GPUs:
            print('-'*20)
            print(f"GPU {gpu.id}: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
        time.sleep(5)  # Print every 5 seconds

threading.Thread(target=print_gpu_memory_usage, daemon=True).start()


# Check if the file exists, if yes, delete it
if os.path.exists(f'{pp}_timing_data.txt'):
    os.remove(f'{pp}_timing_data.txt')

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
        ######
        memory_before = get_gpu_memory()
        cardn =int(j / singlenum)
        p = subprocess.Popen("CUDA_VISIBLE_DEVICES="+str(card[cardn]) + " python3 run.py %d %s %f %d %d"%(lst[totalnum * i + j], project, lr, seed, batch_size), shell=True)
        
        jobs.append(p)
        ######
        memory_after = get_gpu_memory()
        print(f"Memory used by job {j}: {memory_after - memory_before} MiB")

        time.sleep(10)
    for p in jobs:
        p.wait()
p = subprocess.Popen("python3 sum.py %s %d %f %d"%(project, seed, lr, batch_size), shell=True)
p.wait()
subprocess.Popen("python3 watch.py %s %d %f %d"%(project, seed, lr, batch_size),shell=True)            


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