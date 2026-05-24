import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Optional, List
from utils import plot_rank, plot_ipt_graph
import os
import json

class IVONRankAllocator(object):
    """
    IVON LoRA Rank Allocator using Bayesian importance scores.

    Args:
        model: the model that we apply IVON LoRA to.
        lora_r (`int`): The initial rank for each incremental matrix.
        target_rank (`int`): The target average rank of incremental matrix.
        init_warmup (`int`): The steps of initial fine-tuning warmup.
        final_warmup (`int`): The step of final fine-tuning.
        mask_interval (`int`): The time interval between two budget allocations.
        criterion (`str`): Importance scoring criterion.
        total_step (`int`): The total training steps.
        k (`int`): Max ranks adjusted per matrix per step.
        b (`int`): Total ranks adjusted per step.
    """

    def __init__(
            self,
            model,
            lora_r: int,
            target_rank: int,
            init_warmup: int,
            final_warmup: int,
            mask_interval: int,
            criterion: str = "SNR_mean_abs",
            total_step: Optional[int] = None,
            target_total_rank: Optional[int] = None,
            tb_writter=None,
            tb_writter_loginterval: int = 500,
            k: int = 2,
            b: int = 4,
            output_dir: str = None,
            enable_scheduler: bool = False,
    ):
        self.k = k
        self.b = b
        self.initial_b = b
        self.enable_scheduler = enable_scheduler
        self.output_dir = output_dir
        self.criterion = criterion

        self.ave_target_rank = target_rank
        self.target_rank = target_total_rank
        self.lora_init_rank = lora_r#
        self.initial_warmup = init_warmup
        self.final_warmup = final_warmup
        self.mask_interval = mask_interval
        self.total_step = total_step

        self.model = model
        self.score = {}
        self.rank_pattern = {}
        self.get_lora_param_name()

        self.tb_writter = tb_writter
        self.log_interval = tb_writter_loginterval

 
        if self.criterion == "mean":
            self.score_func = self.score_mean
        elif self.criterion == "sigma":
            self.score_func = self.score_sigma
        elif self.criterion == "SNR":
            self.score_func = self.score_SNR
        elif self.criterion == "E_mean_abs":
            self.score_func = self.score_E_mean_abs
        elif self.criterion == "SNR_mean_abs":
            self.score_func = self.score_SNR_mean_abs
        else:
            raise NotImplementedError(f"Unknown criterion: {self.criterion}")


        self.optimizer = None

    def set_optimizer(self, optimizer):

        self.optimizer = optimizer

    def set_total_step(self, total_step: int):
        self.total_step = total_step
        assert self.total_step > self.initial_warmup + self.final_warmup

    def get_rank_pattern(self):
        return self.rank_pattern

    def get_lora_param_name(self):
        self.name_set = set()
        self.total_rank = 0
        self.shape_dict = {}
        for n, p in self.model.named_parameters():
            if "lora_A" in n:
                name_mat = n.replace("lora_A", "%s")
                self.name_set.add(name_mat)
                self.total_rank += p.size(0)
                self.shape_dict[n] = p.shape
            if "lora_B" in n:
                self.shape_dict[n] = p.shape
        self.name_set = list(sorted(self.name_set))
        if self.target_rank is None:
            self.target_rank = self.ave_target_rank * len(self.name_set)



    @staticmethod
    def score_mean(p, p_sigma):
        """|μ|"""
        return torch.abs(p)

    @staticmethod
    def score_sigma(p, p_sigma):
        """1/σ"""
        return 1. / (p_sigma + 1e-8)

    @staticmethod
    def score_SNR(p, p_sigma):
        """μ|/σ"""
        return torch.abs(p) / (p_sigma + 1e-8)

    @staticmethod
    def score_E_mean_abs(p, p_sigma):
        """ E[|θ|]"""
        return p * torch.erf(p / (p_sigma + 1e-8) / np.sqrt(2)) + \
            2 * p_sigma * torch.exp(-p ** 2 / (2 * p_sigma ** 2 + 1e-8)) / np.sqrt(2 * np.pi)

    @staticmethod
    def score_SNR_mean_abs(p, p_sigma):
        """SNR(|θ|)"""
        score = p * torch.erf(p / (p_sigma + 1e-8) / np.sqrt(2)) + \
                2 * p_sigma * torch.exp(-p ** 2 / (2 * p_sigma ** 2 + 1e-8)) / np.sqrt(2 * np.pi)
        return score / torch.sqrt(p_sigma ** 2 + p ** 2 - score ** 2 + 1e-8)

    def update_ipt(self, model):

        if not hasattr(self, "optimizer") or self.optimizer is None:
            print("Warning: IVON optimizer not set")
            return

        try:
            opt_params = self.optimizer.param_groups[0]['params']
            opt_hess = self.optimizer.param_groups[0]['hess']
            wd = self.optimizer.defaults['weight_decay']
            ess = self.optimizer.defaults['ess']
            named_params = {n: p for n, p in model.named_parameters() if p.requires_grad}
            offset = 0

            with torch.no_grad():
                opt_sigma = 1 / torch.sqrt(ess * (opt_hess + wd))
                for i, (n, p) in enumerate(named_params.items()):
                    if not any(lora_name in n for lora_name in ["lora_A", "lora_B", "lora_E"]):
                        continue

                    assert p.shape == opt_params[i].shape, n
                    p = p.detach()
                    numel = p.numel()
                    p_sigma = opt_sigma[offset: offset + numel].view(*p.shape)
                    score = self.score_func(p, p_sigma)
                    score[score != score] = 0  # Remove NaN values
                    self.score[n] = score
                    offset += numel
        except Exception as e:
            print(f"Error in bayesian score update: {e}")

    def calculate_score(self, n, p=None, metric="ipt"):

        if n in self.score:
            return self.score[n]
        else:
  
            return p.abs().detach().clone() if p is not None else torch.tensor(0.0)

    def _combine_ipt(self, ipt_E, ipt_AB):

        ipt_AB = ipt_AB.sum(dim=1, keepdim=False)
        sum_ipt = ipt_E.view(-1) + ipt_AB.view(-1)
        return sum_ipt

    def mask_to_target_rank(self, model, curr_rank):

        is_dict = {}
        combine_dict = {}
        singular_dict = {}

        lora_A_list = []
        lora_B_list = []
        lora_E_list = []

        target_dtype = None
        for n,p in model.named_parameters():
            if 'lora' in n:
                target_dtype = p.dtype
                break


        for n, p in model.named_parameters():
            if "lora_A" in n:
                lora_A_list.append(p)
                ipt_score = self.calculate_score(n, p)
                comb_ipt = torch.mean(ipt_score, dim=1, keepdim=True)
                name_mat = n.replace("lora_A", "%s")
                combine_dict.setdefault(name_mat, []).append(comb_ipt)
            elif "lora_B" in n:
                lora_B_list.append(p)
                ipt_score = self.calculate_score(n, p)
                comb_ipt = torch.mean(ipt_score, dim=0, keepdim=False).view(-1, 1)
                name_mat = n.replace("lora_B", "%s")
                combine_dict.setdefault(name_mat, []).append(comb_ipt)
            elif "lora_E" in n:
                lora_E_list.append(p)
                ipt_score = self.calculate_score(n, p)
                name_mat = n.replace("lora_E", "%s")
                singular_dict[name_mat] = ipt_score


        all_is = []
        for name_mat in combine_dict:
            ipt_E = singular_dict[name_mat]
            ipt_AB = torch.cat(combine_dict[name_mat], dim=1)
            sum_ipt = self._combine_ipt(ipt_E, ipt_AB)
            name_E = name_mat % "lora_E"
            is_dict[name_E] = sum_ipt.view(-1, 1)
            all_is.append(sum_ipt.view(-1))


        top_k_elements = []
        sublist_sizes = []

        for sublist in all_is:
            k = int(min(self.k, sublist.numel() - 1))
            top_k_elements.append(torch.topk(sublist, k, largest=False).values)
            sublist_sizes.append(k)

        flat_top_k_elements = torch.cat(top_k_elements)
        # mask_threshold = torch.topk(flat_top_k_elements, self.b, largest=False).values.max().item()
        smallest_b_elements = torch.topk(flat_top_k_elements, self.b, largest=False).values
        largest_b_elements = torch.topk(flat_top_k_elements, self.b, largest=True).values

        mask_threshold = smallest_b_elements.max().item() 
        mask_threshold_large = largest_b_elements.min().item()  ##

        decrease_idx = torch.topk(flat_top_k_elements, self.b, largest=False).indices
        increase_idx = torch.topk(flat_top_k_elements, self.b, largest=True).indices

        def map_indices(flat_indices, sublist_sizes):
            mapped_sublist_ids = []
            current_offset = 0
            for sublist_id, size in enumerate(sublist_sizes):
                for idx in flat_indices:
                    if current_offset <= idx < current_offset + size:
                        mapped_sublist_ids.append(sublist_id)
                current_offset += size
            return mapped_sublist_ids

        decrease_idx = map_indices(decrease_idx, sublist_sizes)
        increase_idx = map_indices(increase_idx, sublist_sizes)

        def set_nested_attr(obj, attr, value, target_dtype):
            value = value.to(target_dtype)
            attrs = attr.split('.')
            for attr_name in attrs[:-1]:
                obj = getattr(obj, attr_name)
            setattr(obj, attrs[-1], value)

        num_matrix = len(lora_A_list)
        lora_E_name_map = {p: name for name, p in model.named_parameters() if "lora_E" in name}

        # 减少秩
        for i in range(num_matrix):
            if i in decrease_idx:
                matrix_A = lora_A_list[i]
                matrix_B = lora_B_list[i]
                matrix_E = lora_E_list[i]

                with torch.no_grad():
                    matrix_E_name = lora_E_name_map[matrix_E]
                    matrix_A_name = matrix_E_name.replace("lora_E", "lora_A")
                    matrix_B_name = matrix_E_name.replace("lora_E", "lora_B")

                    importance_scores = is_dict[matrix_E_name]
                    below_threshold_indices = (importance_scores <= mask_threshold).nonzero(as_tuple=True)[0]
                    below_threshold_scores = importance_scores[below_threshold_indices].squeeze()

                    num_to_remove = min(sublist_sizes[i], below_threshold_scores.numel())
                    if num_to_remove > 0:
                        removal_indices = torch.topk(
                            below_threshold_scores,
                            num_to_remove,
                            largest=False
                        ).indices
                        removal_indices = below_threshold_indices[removal_indices]
                    else:
                        removal_indices = torch.tensor([], dtype=torch.long, device=importance_scores.device)

                    keep_indices = torch.arange(importance_scores.numel(), device=importance_scores.device)
                    keep_indices = torch.tensor(
                        [idx for idx in keep_indices if idx not in removal_indices],
                        dtype=torch.long,
                        device=importance_scores.device
                    )


                    pruned_matrix_A = torch.index_select(matrix_A, 0, keep_indices).to(target_dtype)
                    pruned_matrix_B = torch.index_select(matrix_B, 1, keep_indices).to(target_dtype)
                    pruned_matrix_E = torch.index_select(matrix_E, 0, keep_indices).to(target_dtype)


                    if matrix_E_name in self.score:
                        self.score[matrix_E_name] = torch.index_select(self.score[matrix_E_name], 0, keep_indices)
                    if matrix_A_name in self.score:
                        self.score[matrix_A_name] = torch.index_select(self.score[matrix_A_name], 0, keep_indices)
                    if matrix_B_name in self.score:
                        self.score[matrix_B_name] = torch.index_select(self.score[matrix_B_name], 1, keep_indices)


                    pruned_matrix_A = torch.nn.Parameter(pruned_matrix_A).to(target_dtype)
                    pruned_matrix_B = torch.nn.Parameter(pruned_matrix_B).to(target_dtype)
                    pruned_matrix_E = torch.nn.Parameter(pruned_matrix_E).to(target_dtype)

                    lora_A_list[i] = pruned_matrix_A
                    lora_B_list[i] = pruned_matrix_B
                    lora_E_list[i] = pruned_matrix_E


                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param is matrix_A:
                            set_nested_attr(model, name, pruned_matrix_A,target_dtype)
                        elif param is matrix_B:
                            set_nested_attr(model, name, pruned_matrix_B,target_dtype)
                        elif param is matrix_E:
                            set_nested_attr(model, name, pruned_matrix_E,target_dtype)


        for i in increase_idx:
            matrix_A = lora_A_list[i]
            matrix_B = lora_B_list[i]
            matrix_E = lora_E_list[i]


            with torch.no_grad():
                new_vector = torch.randn(matrix_A.size(1), device=matrix_A.device, dtype=matrix_A.dtype,requires_grad=True)
                new_vector = new_vector - matrix_A.T @ (matrix_A @ new_vector)
                new_vector = new_vector / (new_vector.norm() + 1e-6)
                new_matrix_A = torch.cat([matrix_A, new_vector.unsqueeze(0)], dim=0).to(target_dtype)
                new_matrix_A = torch.nn.Parameter(new_matrix_A)
                lora_A_list[i] = new_matrix_A

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param is matrix_A:
                        set_nested_attr(model, name, new_matrix_A,target_dtype)


            with torch.no_grad():
                new_vector = torch.randn(matrix_B.size(0), device=matrix_B.device, dtype=matrix_B.dtype,requires_grad=True)
                new_vector = new_vector - matrix_B @ (matrix_B.T @ new_vector)
                new_vector = new_vector / (new_vector.norm() + 1e-6)
                new_matrix_B = torch.cat([matrix_B, new_vector.unsqueeze(1)], dim=1).to(target_dtype)
                new_matrix_B = torch.nn.Parameter(new_matrix_B)
                lora_B_list[i] = new_matrix_B

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param is matrix_B:
                        set_nested_attr(model, name, new_matrix_B,target_dtype)

            with torch.no_grad():
                new_scalar = torch.tensor([min(matrix_E.view(-1).abs().min().item(), 1e-13)],
                                          device=matrix_E.device, dtype=matrix_E.dtype,requires_grad=True)
                new_matrix_E = torch.cat([matrix_E, new_scalar.unsqueeze(0)], dim=0).to(target_dtype)
                new_matrix_E = torch.nn.Parameter(new_matrix_E)
                lora_E_list[i] = new_matrix_E

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param is matrix_E:
                        set_nested_attr(model, name, new_matrix_E,target_dtype)


        if self.output_dir:
            ipt_dir = os.path.join(self.output_dir, "ipt_scores")
            os.makedirs(ipt_dir, exist_ok=True)

            ipt_score_path = os.path.join(ipt_dir, f"step_{self.global_step}.json")
            with open(ipt_score_path, "w") as file:
                all_is_serializable = [item.tolist() if isinstance(item, torch.Tensor) else item for item in all_is]
                json.dump(all_is_serializable, file)


        for n, p in model.named_parameters():
            if "lora_E" in n:
                ranknum = (p != 0.0).sum().item()
                self.rank_pattern[n] = ranknum

        if self.output_dir:
            rank_distribution_dir = os.path.join(self.output_dir, "rank_plots")
            os.makedirs(rank_distribution_dir, exist_ok=True)
            image_path = os.path.join(rank_distribution_dir, f"step_{self.global_step}.png")

            plotting_global_max = max(10, self.lora_init_rank * 2)
            model_start = next(iter(self.rank_pattern)).split(".")[0]
            plot_rank(self.rank_pattern, image_path, 1, plotting_global_max, model_start)

        if hasattr(self, 'optimizer') and self.optimizer is not None:

            trainable_params = [p.to(torch.float32) for p in model.parameters() if p.requires_grad]

            self.optimizer.param_groups[0]['params'] = trainable_params##

            total_numel = sum(p.numel() for p in trainable_params)
            device = trainable_params[0].device

            self.optimizer.param_groups[0]['hess'] = torch.full(
                (total_numel,),
                self.optimizer.defaults['hess_init'],
                device=device,
                dtype=torch.float32
            )
            self.optimizer.param_groups[0]['numel'] = total_numel

            print(f"total para: {total_numel}")
            #mask_threshold = torch.tensor(mask_threshold,dtype=target_dtype)
        return mask_threshold, mask_threshold_large

    def update_and_mask(self, model, global_step):
        self.global_step = global_step
        mask_threshold = None
        mask_threshold_large = None

        if global_step < self.total_step - self.final_warmup:
            self.update_ipt(model)

            if self.enable_scheduler:
                self._b_scheduler(global_step)

            if (global_step > self.initial_warmup and
                    (global_step - self.initial_warmup) % self.mask_interval == 0 and
                    self.b > 0):
                print(f"[IVON LoRA] Now masking, b={self.b}")
                mask_threshold, mask_threshold_large = self.mask_to_target_rank(model, 0)

        return 0, mask_threshold, mask_threshold_large

    def _b_scheduler(self, global_step):
        initial_b = self.initial_b
        final_b = 0
        total_step = self.total_step
        progress = (global_step - self.initial_warmup) / (total_step - self.final_warmup - self.initial_warmup)
        progress = min(max(progress, 0), 1)
        mul_coeff = progress ** 3
        self.b = round(initial_b + (final_b - initial_b) * mul_coeff)


def compute_orth_regu(model, regu_weight=0.1):
    regu_loss, num_param = 0., 0
    for n, p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            para_cov = p @ p.T if "lora_A" in n else p.T @ p
            I = torch.eye(*para_cov.size(), out=torch.empty_like(para_cov))
            I.requires_grad = False
            regu_loss += torch.norm(para_cov - I, p="fro")
            num_param += 1
    return regu_weight * regu_loss / num_param
