# transformer_solver/trainer.py
import torch
from tqdm import tqdm
import os
import time
from torch.utils.data import DataLoader
from tensordict import TensorDict

from common.utils.common import TimeEstimator, clip_grad_norms, unbatchify
from .model import PocatModel
from .pocat_env import PocatEnv
from .pocat_dataset import PocatDataset
from common.pocat_visualizer import print_and_visualize_one_solution

from common.pocat_classes import Battery, Load, PowerIC, LDO, BuckConverter
from common.pocat_defs import PocatConfig, NODE_TYPE_IC
from common.config_loader import load_configuration_from_file


def tensordict_collate_fn(batch):
    """
    DataLoader로부터 받은 TensorDict 샘플 리스트를 하나의 배치 TensorDict로 쌓습니다.
    """
    return torch.stack(batch, dim=0)

def cal_model_size(model, log_func):
    param_count = sum(param.nelement() for param in model.parameters())
    buffer_count = sum(buffer.nelement() for buffer in model.buffers())
    log_func(f'Total number of parameters: {param_count}')
    log_func(f'Total number of buffer elements: {buffer_count}')

class PocatTrainer:
    def __init__(self, args, env: PocatEnv, device: str):
        self.args = args
        self.env = env
        self.device = device
        
        self.model = PocatModel(**args.model_params).to(self.device)
        cal_model_size(self.model, args.log)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(args.optimizer_params['optimizer']['lr']),
            weight_decay=float(args.optimizer_params['optimizer'].get('weight_decay', 0)),
        )
        
        if args.optimizer_params['scheduler']['name'] == 'MultiStepLR':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=args.optimizer_params['scheduler']['milestones'],
                gamma=args.optimizer_params['scheduler']['gamma']
            )
        else:
            raise NotImplementedError
            
        self.start_epoch = 1

        if args.load_path is not None:
            args.log(f"Loading model checkpoint from: {args.load_path}")
            checkpoint = torch.load(args.load_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        if not args.test_only:
            train_dataset = PocatDataset(
                generator=self.env.generator,
                steps_per_epoch=args.trainer_params['train_step']
            )
            
            num_workers = os.cpu_count() // 2 if os.cpu_count() else 4
            args.log(f"데이터 로딩에 {num_workers}개의 CPU 코어를 사용합니다.")

            self.train_dataloader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=tensordict_collate_fn
            )

        self.time_estimator = TimeEstimator(log_fn=args.log)

    def run(self):
        args = self.args
        self.time_estimator.reset(self.start_epoch)
        
        if args.test_only:
            self.test()
            return

        for epoch in range(self.start_epoch, args.trainer_params['epochs'] + 1):
            args.log('=================================================================')
            
            self.model.train()
            
            train_pbar = tqdm(self.train_dataloader, 
                              desc=f"Epoch {epoch}/{args.trainer_params['epochs']}", 
                              ncols=140)
            
            total_loss = 0.0
            total_cost = 0.0
            step_count = 0

            for td in train_pbar:
                step_count += 1
                self.optimizer.zero_grad()
                
                td = td.to(self.device)
                
                # 💡 --- 핵심 수정 ---
                # model 호출 시 불필요한 status_msg 인자를 완전히 제거했습니다.
                model_start_time = time.time()
                out = self.model(td, decode_type='sampling', pbar=train_pbar, log_fn=args.log)
                model_time = time.time() - model_start_time
                
                bwd_start_time = time.time()
                num_starts = self.env.generator.num_loads 
                reward = out["reward"].view(num_starts, -1)
                log_likelihood = out["log_likelihood"].view(num_starts, -1)
                
                advantage = reward - reward.mean(dim=0, keepdims=True)
                loss = -(advantage * log_likelihood).mean()
                loss.backward()
                bwd_time = time.time() - bwd_start_time

                clip_grad_norms(self.optimizer.param_groups, 1.0)
                self.optimizer.step()
                
                best_reward, _ = reward.max(dim=0)
                current_cost = -best_reward.mean().item()

                total_loss += loss.item()
                total_cost += current_cost
                
                train_pbar.set_postfix({
                    'Loss': f'{total_loss/step_count:.4f}',
                    'Cost': f'${total_cost/step_count:.2f}',
                    'T_Model': f'{model_time:.2f}s',
                    'T_Bwd': f'{bwd_time*1000:.0f}ms'
                })
            
            final_desc = f"Epoch {epoch}/{args.trainer_params['epochs']} | Done"
            train_pbar.set_description(final_desc)
            args.log(final_desc)

            self.scheduler.step()
            self.time_estimator.print_est_time(epoch, args.trainer_params['epochs'])
            
            if (epoch % args.trainer_params['model_save_interval'] == 0) or (epoch == args.trainer_params['epochs']):
                save_path = os.path.join(args.result_dir, f'epoch-{epoch}.pth')
                args.log(f"Saving model at epoch {epoch} to {save_path}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, save_path)

        args.log(" *** Training Done *** ")

    @torch.no_grad()
    def test(self):
        args = self.args
        args.log("==================== INFERENCE START ====================")
        self.model.eval()

        td = self.env.reset(batch_size=1)
        
        out = self.model(td, decode_type='greedy')

        num_starts = self.env.generator.num_loads
        reward = unbatchify(out["reward"], num_starts)
        actions = unbatchify(out["actions"], num_starts)

        best_reward, best_idx = reward.max(dim=1)
        best_action_sequence = actions[0, best_idx.item()]
        final_cost = -best_reward.item()

        args.log(f"Generated Power Tree Cost: ${final_cost:.4f}")
        
        self.visualize_result(best_action_sequence, final_cost)


    def visualize_result(self, actions, cost):
        config = self.env.generator.config
        battery = Battery(**config.battery)
        constraints = config.constraints
        loads = [Load(**ld) for ld in config.loads]
        
        expanded_ic_configs = self.env.generator.config.available_ics
        
        candidate_ics = []
        for ic_data in expanded_ic_configs:
            # 💡 --- 핵심 수정 ---
            # IC 타입에 따라 정확한 클래스의 필드만 남기도록 필터링 로직을 강화합니다.
            ic_type = ic_data.get('type')
            if ic_type == 'LDO':
                valid_keys = set(LDO.__dataclass_fields__.keys()) | set(PowerIC.__dataclass_fields__.keys())
                ic_data_filtered = {k: v for k, v in ic_data.items() if k in valid_keys}
                candidate_ics.append(LDO(**ic_data_filtered))
            elif ic_type == 'Buck':
                valid_keys = set(BuckConverter.__dataclass_fields__.keys()) | set(PowerIC.__dataclass_fields__.keys())
                ic_data_filtered = {k: v for k, v in ic_data.items() if k in valid_keys}
                candidate_ics.append(BuckConverter(**ic_data_filtered))

        node_names = config.node_names
        
        active_edges = []
        used_ic_names = set()
        for action in actions:
            child_idx, parent_idx = action[0].item(), action[1].item()
            if parent_idx >= len(node_names) or child_idx >= len(node_names): continue
            child_name = node_names[child_idx]
            parent_name = node_names[parent_idx]
            
            active_edges.append((parent_name, child_name))
            
            if config.node_types[parent_idx] == NODE_TYPE_IC:
                 used_ic_names.add(parent_name)

        solution = {
            "cost": cost,
            "used_ic_names": list(used_ic_names),
            "active_edges": active_edges
        }
        
        print("\n--- Generated Power Tree (Transformer) ---")
        
        print_and_visualize_one_solution(
            solution=solution, 
            candidate_ics=candidate_ics,
            loads=loads, 
            battery=battery, 
            constraints=constraints, 
            solution_index=1
        )