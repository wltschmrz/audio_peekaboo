import rp
import src.peekaboo as peekaboo
from src.peekaboo import run_peekaboo
import torch
import os

from huggingface_hub import login
login(token="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")  # Hugging Face 토큰으로 로그인

# CUDA 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# CUDA 캐시 비우기
torch.cuda.empty_cache()

# 사운드 소스 도메인으로 변경 (예시 URL)
results = run_peekaboo(
    'water drops',
    "extended_water_drops.wav"  # 오디오 파일 URL로 변경
)

# pip uninstall torchvision  # 이미지 관련 라이브러리 제거
# pip install audioldm torchaudio librosa  # AudioLDM 및 torchaudio 설치
 