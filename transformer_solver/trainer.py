# trainer.py

import torch
from tqdm import tqdm
import os

from common.utils.common import TimeEstimator, clip_grad_norms, unbatchify
from .model import PocatModel
from .pocat_env import PocatEnv
from common.pocat_visualizer import print_and_visualize_one_solution

# 💡 수정된 import 구문
from common.pocat_classes import Battery, LDO, BuckConverter, Load
from common.pocat_defs import PocatConfig, NODE_TYPE_IC # <-- NODE_TYPE_IC를 여기서 가져옵니다.
from common.config_loader import load_configuration_from_file




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
            # 💡 1. tqdm의 range를 1부터 시작하도록 변경하여 스텝 번호를 맞춥니다.
            train_pbar = tqdm(range(1, args.trainer_params['train_step'] + 1), 
                              desc=f"Epoch {epoch}/{args.trainer_params['epochs']}", 
                              ncols=100) # 진행률 표시줄의 너비를 고정
            
            total_loss = 0.0
            total_cost = 0.0

            for step in train_pbar:
                self.optimizer.zero_grad()
                td = self.env.reset(
                    batch_size=args.batch_size
                )
                out = self.model(td, self.env)
                
                num_starts = self.env.generator.num_loads
                # reward와 log_likelihood를 (탐색 횟수, 배치 크기) 형태로 변경합니다.
                reward = out["reward"].view(num_starts, -1)
                log_likelihood = out["log_likelihood"].view(num_starts, -1)
                
                # [핵심 수정] 
                # 1. 평균 보상을 기준으로 advantage를 계산합니다.
                #    이제 모든 탐색 결과가 자신의 보상과 전체 평균을 비교하게 됩니다.
                advantage = reward - reward.mean(dim=0, keepdims=True)
                
                # 2. advantage와 모든 log_likelihood를 사용하여 손실을 계산합니다.
                #    'best'가 아닌 모든 결과를 학습에 반영합니다.
                loss = -(advantage * log_likelihood).mean()

                loss.backward()

                
                clip_grad_norms(self.optimizer.param_groups, 1.0)
                self.optimizer.step()
                
                best_reward, _ = reward.max(dim=0) # dim=0으로 수정 (탐색 결과 중 최고)
                current_cost = -best_reward.mean().item()

                total_loss += loss.item()
                total_cost += current_cost
                
                train_pbar.set_postfix({
                    'Loss': f'{total_loss/step:.4f}',
                    'Cost': f'${total_cost/step:.2f}'
                })

            self.scheduler.step()
            self.time_estimator.print_est_time(epoch, args.trainer_params['epochs'])
            
            # 💡 모델 저장 로직 (기존과 동일)
            if (epoch % args.trainer_params['model_save_interval'] == 0) or (epoch == args.trainer_params['epochs']):
                args.log(f"Saving model at epoch {epoch}...")
                # ... (저장 코드) ...

        args.log(" *** Training Done *** ")

    @torch.no_grad()
    def test(self):
        """저장된 모델을 불러와 Power Tree를 생성하고 결과를 시각화합니다."""
        args = self.args
        args.log("==================== INFERENCE START ====================")
        self.model.eval()

        td = self.env.reset(batch_size=64)
        out = self.model(td, self.env)

        num_starts = self.env.generator.num_loads
        reward = unbatchify(out["reward"], num_starts)
        actions = unbatchify(out["actions"], num_starts)

        best_reward, best_idx = reward.max(dim=1)
        best_action_sequence = actions[0, best_idx.item()]
        final_cost = -best_reward.item()

        args.log(f"Generated Power Tree Cost: ${final_cost:.4f}")
        
        # 💡 2. 시각화 함수 호출
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