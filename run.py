import os
import sys
import time
import statistics
import cv2
import numpy as np
import onnxruntime as rt
import multiprocessing as mp
from main import ESRGAN

providers = rt.get_available_providers()
if 'TensorrtExecutionProvider' in providers:
    providers.remove('TensorrtExecutionProvider')
rt.set_default_logger_severity(4)
sess_options = rt.SessionOptions()
sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
if len(providers) == 1 and 'CPUExecutionProvider' in providers:
    use_num_cpus = mp.cpu_count()-1
    sess_options.intra_op_num_threads = int(use_num_cpus/3)
    print(f"CPU mode with providers {providers}")
elif 'CoreMLExecutionProvider' in providers:
    use_num_cpus = mp.cpu_count()-1
    sess_options.intra_op_num_threads = int(use_num_cpus/3)
    print(f"CoreML mode with providers {providers}")
elif 'CUDAExecutionProvider' in providers:
    use_num_cpus = 1
    sess_options.intra_op_num_threads = 1
    print(f"CUDA mode with providers {providers}")

model_path = 'TGHQFace8x_500k-fp32.onnx'
sess_upsk = rt.InferenceSession(model_path, sess_options, providers=providers)
img = cv2.imread('D:/Projects/ESRGAN-ONNX/test_face.png', cv2.IMREAD_COLOR)
img_array = np.array(img)
print('Init....')
for tile_size in [32,48,64,96,128,158,192]:
    print('Tile size=',tile_size)
    model = ESRGAN(model_path, sess_upsk, tile_size, scale=4)
    start_time_0 = time.time()
    time_diffs = []
    reps = 9
    for i in range(reps):
        start_time = time.time()
        result = model.get(img_array)
        time_diff = time.time() - start_time
        time_diffs.append(time_diff)
        #print(f'Iteration time cost: {time.time()-start_time:.4f}s')
    median_time = statistics.median(time_diffs)
    print(f'Median processing time: {median_time:.4f}s')

cv2.imwrite('test_face_.png', result)
