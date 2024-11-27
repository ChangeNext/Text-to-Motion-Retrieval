import os
import torch
from dataset.data import (
    AMASSMotionLoader, 
    Normalizer, 
    TextMotionDataset, 
    VQMotionDataset
)
from dataset.text import SentenceEmbeddings
from src.model.model import load_TeMoLLM
from dataset.data import read_split, load_annotations
from dataset.collate import collate_x_dict
from tqdm.auto import tqdm
import pickle as pkl
import src.utils.utils_model as utils_model

model_dir = "/data/motion/TextMotionRetrieval/TMR_LLM/result/train/32_0.85_32_512_Qwen2-1.5B_True_20240927_1509_best"
logger = utils_model.get_logger(model_dir)
temollm = load_TeMoLLM(model_dir, logger)

nomalizer = Normalizer(base_dir="./dataset/stats/humanml3d/guoh3dfeats", eps=1e-12)
motion_loader = AMASSMotionLoader(base_dir="./dataset/motions/guoh3dfeats", fps=20.0, normalizer=nomalizer, disable= False, nfeats=263)
cnn_motion_loader = VQMotionDataset(unit_length=2**2)

# emb_output_path = 
annotations = load_annotations("./dataset/annotations/humanml3d")
keyids = read_split("./dataset/annotations/humanml3d", split="test")

motion_paths = [] 
keyid_lists = [] 
start_list = []
end_list = []

for keyid in keyids:
    keyid_lists.append(keyid)
    motion_paths.append(annotations[keyid]["path"])
    start_list.append(annotations[keyid]['annotations'][0]["start"])
    end_list.append(annotations[keyid]['annotations'][0]["end"])

output_data = {"paths" : [], "embeddings" : [], "key_id" : []}
with torch.no_grad():
    for motion_path, key_id, start, end in tqdm(zip(motion_paths, keyid_lists, start_list, end_list)):
        temollm.model.eval()
        motion = motion_loader(path = motion_path, start=start, end=end, split="all")
        motion_cnn = cnn_motion_loader(keyid)
        motion_cnn = motion_cnn.unsqueeze(0)
        motion_cnn = motion_cnn.cuda()
        motion_x_dict = collate_x_dict([motion], device="cuda")

        batch = {"motion_x_dict": motion_x_dict, "cnn_motion": motion_cnn}
        
        motion_embs = temollm.model.get_motion_embs_cross_attn(motion_x_dict, motion_cnn)
        motion_embs = motion_embs[0, 0, :].cpu().detach()
        output_data['key_id'].append(key_id)
        output_data['paths'].append(motion_path)
        output_data["embeddings"].append(motion_embs)

path = os.path.join(model_dir, "amass_embeddings_all.pkl")
print(path)
with open(path, "wb") as f:
        pkl.dump(output_data, f)