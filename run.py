import os
import sys
import time
import statistics
from main import ESRGAN

using_model_path = 'Face-Super-Resolution_90000_G.onnx'
print('Init....')
for tile_size in [32,48,64,128]:
    print('Tile size=',tile_size)
    model = ESRGAN(using_model_path, tile_size, scale=4)
    start_time_0 = time.time()
    time_diffs = []
    reps = 16
    for i in range(reps):
        start_time = time.time()
        result = model.get_result('test_face.png')
        time_diff = time.time() - start_time
        time_diffs.append(time_diff)
        #print(f'Iteration time cost: {time.time()-start_time:.4f}s')
    median_time = statistics.median(time_diffs)
    print(f'Median processing time: {median_time:.4f}s')

result.save('test_face_.png')
