from sae.config import TrainerConfig
from sae.sae import load_jump_relu_sae_from_hub
from datasets import load_dataset
import os
from sae.trainer import JumpReLUSaeTrainer
from torch.utils.data import Dataset
from sae.sae import JumpReLUSAE
os.environ["http_proxy"] = "http://127.0.0.1:7891"
os.environ["https_proxy"] = "http://127.0.0.1:7891"

#cfg
cfg = TrainerConfig(
    hook_name='blocks.20.hook_resid_post',
    name = '8_6',
    save_dir = '/home/yyz/jumpReluSAE/test_dir',
    wandb_project='jumprelu'
)

#sae
sae = load_jump_relu_sae_from_hub(repo_id='google/gemma-scope-2b-pt-res', filename='layer_20/width_16k/average_l0_71/params.npz')

#dataset
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return sample

raw_dataset = load_dataset('UUUUUUZ/python-code')
dataset = CustomDataset(raw_dataset['train']['text'])

#hooked transformer
from transformers import GemmaTokenizer, AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

model_name = "/home/yyz/.cache/huggingface/hub/models--google--gemma-2-2b/snapshots/0738188b3055bc98daf0fe7211f0091357e5b979"
t_tokenizer = AutoTokenizer.from_pretrained(model_name, device='cuda')
t_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='cuda'
)
# t_tokenizer = t_tokenizer.to(device)
t_model = t_model.to('cuda')
model = HookedTransformer.from_pretrained(
    "google/gemma-2-2b",
    hf_model=t_model,
    device='cuda',
    n_devices=1,
    tokenizer=t_tokenizer,
    center_unembed=False
)

trainer = JumpReLUSaeTrainer(cfg, dataset, sae, model)
trainer.fit()
