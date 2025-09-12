# transformer_solver/pocat_dataset.py
import torch
from torch.utils.data import Dataset
from .pocat_generator import PocatGenerator
from tensordict import TensorDict

class PocatDataset(Dataset):
    """
    미리 초기화된 PocatGenerator를 사용하여 데이터 샘플을 생성하는 Dataset 클래스입니다.
    """
    def __init__(self, generator: PocatGenerator, steps_per_epoch: int):
        """
        데이터셋을 초기화합니다.
        
        Args:
            generator (PocatGenerator): 이미 전처리가 완료된 PocatGenerator 인스턴스입니다.
            steps_per_epoch (int): 한 에폭 당 생성할 스텝(배치)의 수입니다.
        """
        super().__init__()
        # 💡 각 워커는 메인 프로세스에서 생성된 generator 객체의 복사본을 사용합니다.
        #    무거운 전처리(__init__)는 더 이상 여기서 반복되지 않습니다.
        self.generator = generator
        self.total_samples = steps_per_epoch

    def __len__(self) -> int:
        """한 에폭에 포함된 전체 샘플(배치) 수를 반환합니다."""
        return self.total_samples

    def __getitem__(self, idx: int) -> TensorDict:
        """
        미리 계산된 텐서 템플릿을 사용하여 하나의 데이터 샘플(TensorDict)을 생성합니다.
        """
        # __call__을 호출하여 배치 크기 1짜리 텐서를 생성하고, 배치 차원을 제거합니다.
        return self.generator(batch_size=1).squeeze(0)