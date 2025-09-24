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


def update_progress(pbar, metrics):
    if pbar is None:
        return
    pbar.set_postfix({
        "Loss": f"{metrics['Loss']:.4f}",
        "Avg Cost": f"${metrics['Avg Cost']:.2f}",
        "Min Cost": f"${metrics['Min Cost']:.2f}",
        "T_Reset": f"{metrics['T_Reset']:.0f}ms",
    }, refresh=False)
    pbar.update(1)


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

        self.eval_batch_size = getattr(args, "eval_batch_size", 128)
        with torch.no_grad():
            self._eval_td_fixed = self.env.reset(batch_size=self.eval_batch_size).clone()
        self.best_eval_bom = float("inf")

    def run(self):
        args = self.args
        self.time_estimator.reset(self.start_epoch)
        
        if args.test_only:
            self.test()
            return

        for epoch in range(self.start_epoch, args.trainer_params['epochs'] + 1):
            args.log('=================================================================')
            
            self.model.train()
            
            total_steps = args.trainer_params['train_step']
            train_pbar = tqdm(
                total=total_steps,
                desc=f"Epoch {epoch}",
                dynamic_ncols=True,
                leave=False,
                miniters=1,
                mininterval=0.1,
                smoothing=0.1,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
            )
            
            total_loss = 0.0
            total_cost = 0.0
            min_epoch_cost = float('inf') # 💡 **[변경 1]** 에포크 내 최소 비용을 기록할 변수 추가

            for step in range(1, total_steps + 1):
                self.optimizer.zero_grad()
                
                reset_start_time = time.time()
                td = self.env.reset(batch_size=args.batch_size)
                reset_time = time.time() - reset_start_time
                
                model_start_time = time.time()
                # --- 👇 [핵심] log 함수를 모델에 전달 ---
                out = self.model(td, self.env, decode_type='sampling', pbar=train_pbar,
                                     status_msg=None, log_fn=args.log,
                                     log_idx=args.log_idx, log_mode=args.log_mode)
                model_time = time.time() - model_start_time
                
                bwd_start_time = time.time()
                num_starts = self.env.generator.num_loads
                reward = out["reward"].view(num_starts, -1)
                log_likelihood = out["log_likelihood"].view(num_starts, -1)
                
                advantage = reward - reward.mean(dim=0, keepdims=True)
                loss = -(advantage * log_likelihood).mean()
                loss.backward()
                
                # 그래디언트 클리핑 (옵션)
                max_norm = float(self.args.optimizer_params.get('max_grad_norm', 0))
                if max_norm > 0:
                    clip_grad_norms(self.optimizer.param_groups, max_norm=max_norm)

                # 가중치 업데이트
                self.optimizer.step()

                bwd_time = time.time() - bwd_start_time

                logging.debug(
                    "Epoch %d step %d reset=%.3fms model=%.3fms backward=%.3fms",
                    epoch,
                    step,
                    reset_time * 1000,
                    model_time * 1000,
                    bwd_time * 1000,
                )

                best_reward_per_instance = reward.max(dim=0)[0]
                
                # 💡 **[변경 2]** 현재 배치의 평균 비용과 최소 비용 계산
                avg_cost = -best_reward_per_instance.mean().item()
                min_batch_cost = -best_reward_per_instance.max().item()
                min_epoch_cost = min(min_epoch_cost, min_batch_cost)


                total_loss += loss.item()
                total_cost += avg_cost
                
                update_progress(
                    train_pbar,
                    {
                        "Loss": loss.item(),
                        "Avg Cost": total_cost / step,
                        "Min Cost": min_epoch_cost,
                        "T_Reset": reset_time * 1000,
                    },
                )

            train_pbar.close()

            epoch_summary = (
                f"Epoch {epoch}/{args.trainer_params['epochs']} | "
                f"Loss {total_loss / total_steps:.4f} | "
                f"Avg Cost ${total_cost / total_steps:.2f} | "
                f"Min Cost ${min_epoch_cost:.2f}"
            )
            tqdm.write(epoch_summary)
            args.log(epoch_summary) # 에폭 종료 메시지도 로그에 기록
            val = self.evaluate(epoch)
            self.args.log(f"[Eval] Epoch {epoch} | Avg BOM ${val['avg_bom']:.2f} | Min BOM ${val['min_bom']:.2f}")

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
    def evaluate(self, epoch: int):
        """Greedy decode on a fixed validation set, CSV log, and save best checkpoint by avg BOM."""
        self.model.eval()

        # Rebuild eval TD from the fixed snapshot (same instances every epoch)
        td_eval = self.env._reset(self._eval_td_fixed.clone())

        # Greedy decoding; reuse your model call signature
        out = self.model(
            td_eval, self.env, decode_type='greedy',
            pbar=None, status_msg="Eval",
            log_fn=self.args.log, log_idx=self.args.log_idx, log_mode=self.args.log_mode
        )

        # POMO starts: choose best per instance
        num_starts = self.env.generator.num_loads
        reward = out["reward"].view(num_starts, -1)
        best_reward_per_instance = reward.max(dim=0)[0]

        avg_bom = -best_reward_per_instance.mean().item()
        min_bom = -best_reward_per_instance.max().item()

        # CSV log
        import os, torch
        csv_path = os.path.join(self.result_dir, "val_log.csv")
        header = not os.path.exists(csv_path)
        with open(csv_path, "a", encoding="utf-8") as f:
            if header:
                f.write("epoch,avg_bom,min_bom,decode_type\n")
            f.write(f"{epoch},{avg_bom:.4f},{min_bom:.4f},greedy\n")

        # Save best
        if avg_bom < self.best_eval_bom:
            self.best_eval_bom = avg_bom
            save_path = os.path.join(self.result_dir, "best_cost.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, save_path)
            self.args.log(f"[Eval] ✅ New best avg_bom=${avg_bom:.2f} (min=${min_bom:.2f}) at epoch {epoch} → saved {save_path}")

        return {"avg_bom": avg_bom, "min_bom": min_bom}

    def test(self):
        self.model.eval()
        logging.info("==================== INFERENCE START ====================")

        td = self.env.reset(batch_size=1)
        
        # --- 👇 [핵심 수정 1] POMO 시작 노드 정보 가져오기 ---
        _, start_nodes_idx = self.env.select_start_nodes(td)
        
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

        # --- 👇 [핵심 수정 2] 최적해의 시작 노드 이름 찾기 및 출력 ---
        best_start_node_idx = start_nodes_idx[best_idx].item()
        best_start_node_name = self.env.generator.config.node_names[best_start_node_idx]
        print(f"Generated Power Tree (Best start: '{best_start_node_name}'), Cost: ${final_cost:.4f}")

        action_history = []
        td_sim = self.env._reset(td.clone())

        td_sim.set("action", best_action_sequence[0])
        output_td = self.env.step(td_sim)
        td_sim = output_td["next"]
        
        for action_tensor in best_action_sequence[1:]:
            if td_sim["done"].all(): break
            current_head = td_sim["trajectory_head"].item()
            action_item = action_tensor.item()
            if current_head != BATTERY_NODE_IDX:
                action_history.append((action_item, current_head))
            td_sim.set("action", action_tensor)
            output_td = self.env.step(td_sim)
            td_sim = output_td["next"]

        # --- 👇 [핵심 수정 3] 시각화 함수에 시작 노드 이름 전달 ---
        self.visualize_result(action_history, final_cost, best_start_node_name)


    def visualize_result(self, action_history, final_cost, best_start_node_name):
        """
        [수정됨] graphviz를 사용하고 시작 노드 정보를 포함하여 시각화합니다.
        """
        if self.result_dir is None: return
        os.makedirs(self.result_dir, exist_ok=True)

        node_names = self.env.generator.config.node_names

        dot = Digraph(comment=f"Power Tree Topology - Cost ${final_cost:.4f}")
        dot.attr('node', shape='box', style='rounded')
        
        # --- 👇 [핵심 수정 4] 그래프 제목에 시작 노드 정보 추가 ---
        label_text = f"Best Solution (Started from: {best_start_node_name})\\nCost: ${final_cost:.4f}"
        dot.attr(rankdir='LR', label=label_text, labelloc='t')

        used_node_indices = set()
        for parent_idx, child_idx in action_history:
            used_node_indices.add(parent_idx)
            used_node_indices.add(child_idx)
        
        for node_idx in used_node_indices:
            node_name = node_names[node_idx]
            dot.node(node_name, node_name)
        
        for parent_idx, child_idx in action_history:
            parent_name = node_names[parent_idx]
            child_name = node_names[child_idx]
            dot.edge(parent_name, child_name)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"solution_cost_{final_cost:.4f}_{timestamp}"
        output_path = os.path.join(self.result_dir, filename)
        
        try:
            dot.render(output_path, view=False, format='png', cleanup=True)
            logging.info(f"Power tree visualization saved to {output_path}.png")
        except Exception as e:
            logging.error(f"Failed to render visualization. Is Graphviz installed and in your PATH? Error: {e}")






