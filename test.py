import os
import json
import torch 
from src.model.model import TeMoLLM, TeMoLLM_Config
from src.retrieval import retrieval
from dataset.text import SentenceEmbeddings
from dataset.data import (
    AMASSMotionLoader, 
    Normalizer, 
    TextMotionDataset, 
    VQMotionDataset
)
from collections import namedtuple
from src.model.model import load_TeMoLLM
import src.utils.utils_model as utils_model

nomalizer = Normalizer(base_dir="./dataset/annotations/humanml3d/stats/humanml3d/guoh3dfeats", eps=1e-12)
motion_loader = AMASSMotionLoader(base_dir="./dataset/motions/guoh3dfeats", fps=20.0, normalizer=nomalizer, disable= False, nfeats=263)
sent_emd = SentenceEmbeddings(modelname = "sentence-transformers/all-mpnet-base-v2", path = "./dataset/annotations/humanml3d", preload= True)
cnn_motion_loader = VQMotionDataset(unit_length=2**2)

if __name__ == "__main__":
    
    model_dir = "/data/motion/TextMotionRetrieval/TMR_LLM/result/train/32_0.9_32_512_Qwen2-1.5B_True_20241111_0257"
    basename = os.path.basename(model_dir)
    save_dir = f"./result/test/{basename}"

    os.makedirs(save_dir, exist_ok=True)
    logger = utils_model.get_logger(save_dir)

    temollm = load_TeMoLLM(model_dir, logger)

    nsim_dataset = TextMotionDataset(
        path = './dataset/annotations/humanml3d', 
        split = 'nsim_test',
        max_len = 32,
        tokenizer=temollm.model.tokenizer,
        motion_loader = motion_loader,
        cnn_motion_loader=cnn_motion_loader, 
        text_to_sent_emb = sent_emd,
        preload=False,
        return_dict=True,
        echo=temollm.config.echo
        )

    test_dataset = TextMotionDataset(
        path = './dataset/annotations/humanml3d', 
        split = 'test',
        max_len = 32,
        tokenizer=temollm.model.tokenizer,
        motion_loader = motion_loader,
        cnn_motion_loader=cnn_motion_loader, 
        text_to_sent_emb = sent_emd,
        preload=False,
        return_dict=True,
        echo=temollm.config.echo
        )

    metric = retrieval(protocol="all", dataset=test_dataset, threshold=0.95, model=temollm.model, device="cuda", save_dir = save_dir, train_mode = False, batch_size=137, logger=None, bf16 = temollm.model.bf16, nsim_dataset=nsim_dataset)
    # metric = retrieval(protocol="nsim", dataset=test_dataset, threshold=0.95, model=temollm.model, device="cuda", save_dir = save_dir, train_mode = False, batch_size=137, logger=None, bf16 = temollm.model.bf16, nsim_dataset=nsim_dataset)
    
    