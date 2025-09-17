# transformer_solver/trainer.py
import torch
from tqdm import tqdm
import os
import time # 💡 시간 측정을 위해 time 모듈 추가
from datetime import datetime

import json
import logging

from common.utils.common import TimeEstimator, clip_grad_norms, unbatchify
from .model import PocatModel
from .pocat_env import PocatEnv
from common.pocat_visualizer import print_and_visualize_one_solution

from common.pocat_classes import Battery, LDO, BuckConverter, Load
from common.pocat_defs import PocatConfig, NODE_TYPE_IC
from common.config_loader import load_configuration_from_file
from .pocat_env import BATTERY_NODE_IDX
from graphviz import Digraph



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

        self.result_dir = args.result_dir

        
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
            
            train_pbar = tqdm(range(1, args.trainer_params['train_step'] + 1), 
                              desc=f"Epoch {epoch}/{args.trainer_params['epochs']}", 
                              ncols=140)
            
            total_loss = 0.0
            total_cost = 0.0
            min_epoch_cost = float('inf') # 💡 **[변경 1]** 에포크 내 최소 비용을 기록할 변수 추가

            for step in train_pbar:
                step_start_time = time.time()
                self.optimizer.zero_grad()
                
                base_desc = f"Epoch {epoch} (Step {step})"
                status_message = f"🔄 Env Reset (ing..)"
                
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
                out = self.model(td, self.env, decode_type='sampling', pbar=train_pbar,
                                     status_msg=status_message, log_fn=args.log,
                                     log_idx=args.log_idx, log_mode=args.log_mode)
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

                best_reward_per_instance = reward.max(dim=0)[0]
                
                # 💡 **[변경 2]** 현재 배치의 평균 비용과 최소 비용 계산
                avg_cost = -best_reward_per_instance.mean().item()
                min_batch_cost = -best_reward_per_instance.max().item()
                min_epoch_cost = min(min_epoch_cost, min_batch_cost)


                total_loss += loss.item()
                total_cost += avg_cost
                
                train_pbar.set_postfix({
                    'Loss': f'{total_loss/step:.4f}',
                    'Avg Cost': f'${total_cost/step:.2f}',
                    'Min Cost': f'${min_epoch_cost:.2f}',
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
        self.model.eval()
        logging.info("==================== INFERENCE START ====================")

        # 테스트 환경 리셋 (배치 크기는 1로 고정)
        td = self.env.reset(batch_size=1)
        
        pbar = tqdm(total=1, desc=f"Solving Power Tree (Mode: {self.args.decode_type})")
        out = self.model(td, self.env, decode_type=self.args.decode_type, pbar=pbar, 
                         log_fn=logging.info, log_idx=self.args.log_idx, 
                         log_mode=self.args.log_mode)
        pbar.close()

        reward = out['reward']
        actions = out['actions']
        
        best_idx = reward.argmax()
        final_cost = -reward[best_idx].item()
        best_action_sequence = actions[best_idx]

        print(f"Generated Power Tree Cost: ${final_cost:.4f}")
        # --- 👇 [핵심 로직] 행동 시퀀스를 기반으로 (부모, 자식) 연결 관계 재구성 ---
        action_history = []
        # 테스트 데이터와 동일한 조건으로 시뮬레이션 환경을 리셋
        td_sim = self.env._reset(td.clone())

        # 첫 번째 행동(시작 Load 선택)은 연결 관계를 만들지 않음
        td_sim.set("action", best_action_sequence[0])
        output_td = self.env.step(td_sim)
        td_sim = output_td["next"]
        
        # 두 번째 행동부터 시뮬레이션하며 연결 관계 추적
        for action_tensor in best_action_sequence[1:]:
            # 루프 시작 전에 done 상태를 먼저 확인
            if td_sim["done"].all():
                break

            current_head = td_sim["trajectory_head"].item()
            action_item = action_tensor.item()

            if current_head != BATTERY_NODE_IDX:
                action_history.append((action_item, current_head)) # (부모, 자식)

            td_sim.set("action", action_tensor)
            output_td = self.env.step(td_sim)
            td_sim = output_td["next"]

        self.visualize_result(action_history, final_cost)


    def visualize_result(self, action_history, final_cost):
        """
        [수정됨] graphviz를 직접 사용하여 간단한 Power Tree 토폴로지를 시각화합니다.
        """
        if self.result_dir is None: return
        os.makedirs(self.result_dir, exist_ok=True)

        # 노드 이름을 가져옵니다.
        node_names = self.env.generator.config.node_names

        # Digraph 객체 생성
        dot = Digraph(comment=f"Power Tree Topology - Cost ${final_cost:.4f}")
        dot.attr('node', shape='box', style='rounded')
        dot.attr(rankdir='LR', label=f"Solution Cost: ${final_cost:.4f}", labelloc='t')

        # 모든 노드를 그래프에 추가
        for name in node_names:
            dot.node(name, name)
        
        # (부모, 자식) 연결 관계를 기반으로 엣지 추가
        for parent_idx, child_idx in action_history:
            parent_name = node_names[parent_idx]
            child_name = node_names[child_idx]
            dot.edge(parent_name, child_name)
        
        # 파일로 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"solution_cost_{final_cost:.4f}_{timestamp}"
        output_path = os.path.join(self.result_dir, filename)
        
        try:
            dot.render(output_path, view=False, format='png', cleanup=True)
            logging.info(f"Power tree visualization saved to {output_path}.png")
        except Exception as e:
            logging.error(f"Failed to render visualization. Is Graphviz installed and in your PATH? Error: {e}")











