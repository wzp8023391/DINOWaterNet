# coding:utf-8
import torch
from thop import profile
import time

try:
    from AI_model_iFormer import iFormer_s
except:
    from AI_model_iFormer import iFormer_s

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    inputChannel, numClass = 3, 2
    
    input = torch.randn(2, 3, 1024, 1024).to(device)
    
    model = iFormer_s(numClass = 2).to(device) 
    
    flops, params = profile(model, (input,))  # Parameter
    print('diff_flops: G', flops/1e9, 'diff_params: M', params/1e6)
    
    time_s = time.time()                      # inference time
    result = model(input)
    
    time_e = time.time()
    time_all = time_e - time_s
    print("cost time:", time_all)