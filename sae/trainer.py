from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from transformer_lens import HookedTransformer
from transformers import GemmaTokenizer, AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim
import wandb
import torch
from .config import TrainerConfig
from .sae import JumpReLUSAE
from .sae import StepFunction

class JumpReLUSaeTrainer:
    def __init__(self, cfg: TrainerConfig, dataset: Dataset, sae: JumpReLUSAE, model: HookedTransformer):

        # initiate optimizer
        num_examples = len(dataset)
            #threshold training is seperated
        self.optimizer = optim.Adam(sae.parameters(), lr=cfg.lr)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer, cfg.lr_warmup_steps, num_examples // cfg.batch_size
        )

        self.cfg = cfg
        self.model = model
        self.sae = sae
        self.dataset = dataset

    
    def fit(self):
        wandb.init(
            project=self.cfg.wandb_project, 
            name=self.cfg.name,
        )
        dl = DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )
        num_tokens_in_step = 0
        pbar = tqdm(dl, desc="Training")
        #train
        for epoch in range(self.cfg.epoch):
            for i, batch in enumerate(pbar):
                # print(batch)
                tokens = self.model.to_tokens(batch).to(self.sae.device)
                # print(tokens)
                num_tokens_in_step += tokens.numel()
                # print(num_tokens_in_step)
                with torch.no_grad():
                    logits, cache = self.model.run_with_cache(tokens)
                    # print(cache[self.cfg.hook_name].shape)
                sae_input = cache[self.cfg.hook_name]
                # cal loss
                x_reconstructed, feature_magnitudes = self.sae(sae_input)
                reconstruction_error = sae_input - x_reconstructed
                reconstruction_loss = torch.sum(reconstruction_error**2, dim=-1)
                # print(reconstruction_loss.shape)
                threshold = torch.exp(self.sae.log_threshold)
                l0 = torch.sum(StepFunction.apply(feature_magnitudes, threshold), dim=-1)
                sparsity_loss = self.cfg.sparsity_coefficient * l0
                # print(sparsity_loss.shape)
                total_loss = torch.mean(reconstruction_loss + sparsity_loss)
                # print(total_loss.shape)
                #log to wandb
                if i % 10 == 0:
                    wandb.log({
                        'reconstruction_loss': torch.mean(reconstruction_loss).item(),
                        'sparsity_loss': torch.mean(sparsity_loss).item(),
                        'total_loss': torch.mean(total_loss).item(),
                        'num_tokens_in_step': num_tokens_in_step
                    })

                total_loss.backward()
                self.optimizer.step()

        wandb.finish()
        self.save()

    def save(self):
        torch.save(self.sae.state_dict(), self.cfg.save_dir + '/sae_model.pth')