# utils/utils.py

import os
import sys
import shutil

from .common import TimeEstimator


def copy_all_src(dst_root: str):
    """현재 실행된 소스 코드를 백업합니다."""
    try:
        execution_path = os.path.dirname(os.path.abspath(sys.argv[0]))
        dst_path = os.path.join(dst_root, 'src_backup')

        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        for root, _, files in os.walk(execution_path):
            for file in files:
                if file.endswith('.py'):
                    src_file_path = os.path.join(root, file)
                    if 'site-packages' in src_file_path or 'venv' in src_file_path:
                        continue

                    relative_path = os.path.relpath(src_file_path, execution_path)
                    dst_file_path = os.path.join(dst_path, relative_path)

                    os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)
                    shutil.copy(src_file_path, dst_file_path)
        print(f"Source code backed up to: {dst_path}")
    except Exception as e:
        print(f"Could not back up source code: {e}")
