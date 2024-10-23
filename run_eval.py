import torch
import torch.nn as nn
import numpy as np
import os
from models.seg_model import DilateResUNetCLMem
from evaluate import evaluate_ct, evaluate_mr


########################
##### UDA on mr2ct #####
########################
def uda_on_ct():
    model = DilateResUNetCLMem(n_channels=1, n_classes=5, norm="InstanceNorm")
    model.load_state_dict(torch.load(args.model_path))

    model = model.cuda()
    model.eval()

    with torch.no_grad():
        evaluate_ct(model, use_assd=True)
        

########################
##### UDA on ct2mr #####
########################

def uda_on_mr():
    model = DilateResUNetCLMem(n_channels=1, n_classes=5, norm="InstanceNorm")
    model.load_state_dict(torch.load(args.model_path))

    model = model.cuda()
    model.eval()

    with torch.no_grad():
        evaluate_mr(model, use_assd=True)


if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="ct")
    parser.add_argument("--model_path", type=str, default="xxx.pt")
    args = parser.parse_args()

    if args.target == "ct":
        uda_on_ct()
    elif args.target == "mr":
        uda_on_mr()