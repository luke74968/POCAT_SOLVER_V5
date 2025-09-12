# transformer_solver/trainer.py
import torch
from tqdm import tqdm
import os
import time # 💡 시간 측정을 위해 time 모듈 추가

# 💡 1. 필요한 모듈들을 임포트합니다.
from torch.utils.data import DataLoader
from tensordict import TensorDict


from common.utils.common import TimeEstimator, clip_grad_norms, unbatchify
from .model import PocatModel
from .pocat_env import PocatEnv
from .pocat_dataset import PocatDataset # 💡 2. 새로 만든 Dataset 클래스를 임포트합니다.
from common.pocat_visualizer import print_and_visualize_one_solution

from common.pocat_classes import Battery, LDO, BuckConverter, Load
from common.pocat_defs import PocatConfig, NODE_TYPE_IC
from common.config_loader import load_configuration_from_file

def tensordict_collate_fn(batch):
    """
    DataLoader로부터 받은 TensorDict 샘플 리스트를 하나의 배치 TensorDict로 쌓습니다.
    """
    # batch는 [TensorDict_sample1, TensorDict_sample2, ...] 형태의 리스트입니다.
    return torch.stack(batch, dim=0)

def cal_model_size(model, log_func):
    param_count = sum(param.nelement() for param in model.parameters())
    buffer_count = sum(buffer.nelement() for buffer in model.buffers())
    log_func(f'Total number of parameters: {param_count}')
    log_func(f'Total number of buffer elements: {buffer_count}')

class PocatTrainer:
    # 💡 1. 생성자에서 device 인자를 받도록 수정
    def __init__(self, args, env: PocatEnv, device: str):
        self.args = args
        self.env = env
        self.device = device # 전달받은 device 저장
        
        # 💡 2. CUDA 강제 설정 라인 삭제
        # torch.set_default_tensor_type('torch.cuda.FloatTensor') 
        
        # 💡 3. 모델을 생성 후, 지정된 device로 이동
        self.model = PocatModel(**args.model_params).to(self.device)
        cal_model_size(self.model, args.log)
        
        # 💡 float()으로 감싸서 값을 숫자로 강제 변환합니다.
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

        # 💡 모델 로딩 로직 추가
        if args.load_path is not None:
            args.log(f"Loading model checkpoint from: {args.load_path}")
            checkpoint = torch.load(args.load_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            # 훈련을 이어서 할 경우 optimizer 상태도 불러올 수 있음
            # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # self.start_epoch = checkpoint['epoch'] + 1       
            # 
                # 💡 3. 학습을 위한 Dataset과 DataLoader를 생성합니다.
        if not args.test_only:
            train_dataset = PocatDataset(
                generator=self.env.generator,
                steps_per_epoch=args.trainer_params['train_step']
            )
            
            # os.cpu_count()를 사용해 사용 가능한 코어 수를 파악합니다.
            # 코어의 절반 정도를 사용하는 것이 안전한 시작점입니다.
            num_workers = os.cpu_count() // 2 if os.cpu_count() else 4
            args.log(f"데이터 로딩에 {num_workers}개의 CPU 코어를 사용합니다.")

            # 💡 3. collate_fn에 새로 정의한 top-level 함수를 전달합니다.
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
            
            # 💡 4. 학습 루프를 DataLoader 기준으로 변경합니다.
            train_pbar = tqdm(self.train_dataloader, 
                              desc=f"Epoch {epoch}/{args.trainer_params['epochs']}", 
                              ncols=140)
            
            total_loss = 0.0
            total_cost = 0.0
            step_count = 0


            for td in train_pbar:
                step_count += 1
                self.optimizer.zero_grad()

                # DataLoader가 준비된 배치를 제공하므로, 디바이스로 옮기기만 하면 됩니다.
                td = td.to(self.device)

                base_desc = f"Epoch {epoch} (Step {step_count})"
                status_message = f"🔄 Env Reset (ing..)"
                
                # 진행률 표시줄 로깅은 이제 모델의 forward 패스 내부에서 처리됩니다.
                # 여기서는 상태 메시지를 단순화할 수 있습니다.
                base_desc = f"Epoch {epoch} (Step {step_count})"
                
                # --- 👇 [핵심] tqdm 설명 설정과 동시에 로그 기록 ---
                train_pbar.set_description(f"{base_desc} | {status_message}")
                args.log(train_pbar.desc)

                reset_start_time = time.time()
                td = self.env.reset(batch_size=args.batch_size)
                reset_time = time.time() - reset_start_time
                
                status_message = f"🔄 Env Reset (done)"
                train_pbar.set_description(f"{base_desc} | {status_message}")
                args.log(train_pbar.desc)

                model_start_time = time.time()
                # --- 👇 [핵심] log 함수를 모델에 전달 ---
                out = self.model(td, decode_type='sampling', pbar=train_pbar, status_msg=status_message, log_fn=args.log)
                model_time = time.time() - model_start_time

                status_message += f" | ▶ Encoding (done) | ◀ Decoding (done)"
                status_message += f" | 📉 Loss & Bwd (ing..)"
                train_pbar.set_description(f"{base_desc} | {status_message}")
                args.log(train_pbar.desc)
                
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
                    'T_Reset': f'{reset_time*1000:.0f}ms',
                    'T_Model': f'{model_time:.2f}s',
                    'T_Bwd': f'{bwd_time*1000:.0f}ms'
                })
            
            final_desc = f"Epoch {epoch}/{args.trainer_params['epochs']} | Done"
            train_pbar.set_description(final_desc)
            args.log(final_desc) # 에폭 종료 메시지도 로그에 기록

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


    # ... (test, visualize_result 메소드는 기존과 동일) ...
    @torch.no_grad()
    def test(self):
        args = self.args
        args.log("==================== INFERENCE START ====================")
        self.model.eval()

        td = self.env.reset(batch_size=1) # 테스트는 배치 1로 고정
        
        # --- 👇 [핵심] 테스트 시에는 'greedy' 방식으로 모델 호출 ---
        out = self.model(td, self.env, decode_type='greedy')

        num_starts = self.env.generator.num_loads
        reward = unbatchify(out["reward"], num_starts)
        actions = unbatchify(out["actions"], num_starts)

        best_reward, best_idx = reward.max(dim=1)
        best_action_sequence = actions[0, best_idx.item()]
        final_cost = -best_reward.item()

        args.log(f"Generated Power Tree Cost: ${final_cost:.4f}")
        
        self.visualize_result(best_action_sequence, final_cost)


    def visualize_result(self, actions, cost):
        """모델이 생성한 action_sequence를 기반으로 결과를 시각화합니다."""
        
        # --- 💡 1. config.json을 다시 로드하는 대신, generator의 확장된 config를 사용 ---
        config = self.env.generator.config
        battery = Battery(**config.battery)
        constraints = config.constraints
        loads = [Load(**ld) for ld in config.loads]
        
        # Generator가 동적 복제한 전체 IC 목록(dict)을 가져옴
        expanded_ic_configs = config.available_ics
        
        # 시각화를 위해 dict를 PowerIC 객체로 변환
        candidate_ics = []
        for ic_data in expanded_ic_configs:
            ic_type = ic_data.get('type')
            if ic_type == 'LDO':
                candidate_ics.append(LDO(**ic_data))
            elif ic_type == 'Buck':
                candidate_ics.append(BuckConverter(**ic_data))
        # --- 수정 완료 ---

        node_names = config.node_names
        
        active_edges = []
        used_ic_names = set()
        for action in actions:
            child_idx, parent_idx = action[0].item(), action[1].item()
            child_name = node_names[child_idx]
            parent_name = node_names[parent_idx]
            
            active_edges.append((parent_name, child_name))
            
            if config.node_types[parent_idx] == NODE_TYPE_IC:
                 used_ic_names.add(parent_name)

        solution = {
            "cost": cost,
            "used_ic_names": used_ic_names,
            "active_edges": active_edges
        }
        
        print("\n--- Generated Power Tree (Transformer) ---")
        
        print_and_visualize_one_solution(
            solution=solution, 
            candidate_ics=candidate_ics, # 💡 확장/변환된 IC 리스트 전달
            loads=loads, 
            battery=battery, 
            constraints=constraints, 
            solution_index=1
        )