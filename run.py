import cv2
import time
import statistics
import numpy as np
import onnxruntime as rt
import multiprocessing as mp
from src.ESRGAN_ONNX import ESRGAN
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input', type=str, default=None,
                    help='the path to the input file')
parser.add_argument('--output', type=str, default=None,
                    help='the path to the output file')
parser.add_argument('--prepad', type=int, default=8,
                    help='padding size')
parser.add_argument('--img_size', type=int, default=128,
                    help='an integer representing the size of img_array if no input is provided')
parser.add_argument('--model_path', type=str, default='2x_DigitalFlim_SuperUltraCompact_nf24-nc8_289k_net_g.onnx',
                    help='the path to the ONNX model file')
parser.add_argument('--reps', type=int, default=9,
                    help='Median repetition count')
args = parser.parse_args()

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

sess_upsk = rt.InferenceSession(args.model_path, sess_options, providers=providers)
if not args.input:
    img_array = np.full((args.img_size, args.img_size, 3), (255, 255, 255), dtype=np.uint8)
else:
    img_array = np.array(cv2.imread(args.input, cv2.IMREAD_COLOR))
best_median_time = float('inf')
best_tile_size = None
median_times = []
tile_size=[32,48,64,96,128,160,192]
print('Model', args.model_path, '\nImage size =', img_array.shape)
for tile in tile_size:
    print('Tile size=',tile)
    model = ESRGAN(sess_upsk, tile, args.prepad)
    start_time_0 = time.time()
    time_diffs = []
    for i in range(args.reps):
        start_time = time.time()
        result = model.get(img_array)[0]
        time_diff = time.time() - start_time
        time_diffs.append(time_diff)
        #print(f'Iteration time cost: {time.time()-start_time:.4f}s')
    median_time = statistics.median(time_diffs)
    median_times.append(median_time)
    print(f'Median processing time: {median_time:.4f}s')

    if median_time < best_median_time:
        best_median_time = median_time
        best_tile_size = tile
print(f'Best tile size: {best_tile_size}, Median processing time: {best_median_time:.4f}s')
if args.output:
    cv2.imwrite(args.output, result)