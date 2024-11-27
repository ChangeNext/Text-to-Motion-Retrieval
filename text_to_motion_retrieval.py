from src.model.model import load_TeMoLLM_Retrieval
import src.utils.utils_model as utils_model

model_dir = "/data/motion/TextMotionRetrieval/TMR_LLM/result/train/32_0.85_32_512_Qwen2-1.5B_True_20240927_1509"
logger = utils_model.get_logger(model_dir)
TeMoLLM = load_TeMoLLM_Retrieval(model_dir=model_dir,logger=logger)

prompt = "a person moves backwards then forwards then jumps."

return_outputs = TeMoLLM.Text_Motion_Retrieval(prompt=prompt, max_mot_per_ret=5, echo=True)

print(return_outputs)


