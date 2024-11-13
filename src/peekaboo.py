from typing import Union, List, Optional

import numpy as np
import rp
import torch
import torch.nn as nn
from easydict import EasyDict
from einops import repeat

import utils as ut
import src.audioldm as aldm
from src.bilateralblur_learnabletextures import (BilateralProxyBlur,
                                    LearnableImageFourier,
                                    LearnableImageFourierBilateral,
                                    LearnableImageRaster,
                                    LearnableImageRasterBilateral)
from src.config import default_audioldm_config as cf
from src.audio import wav_to_fbank, TacotronSTFT, read_wav_file


# !!! 알파 채널도 조정가능하게 수정해야함. ----------
def make_learnable_image(height, width, num_channels, foreground=None, bilateral_kwargs=None, representation='fourier'):
    "이미지의 파라미터화 방식을 결정하여 학습 가능한 이미지를 생성."
    bilateral_kwargs = bilateral_kwargs or {}
    bilateral_blur = BilateralProxyBlur(foreground, **bilateral_kwargs)
    if representation == 'fourier bilateral':
        return LearnableImageFourierBilateral(bilateral_blur, num_channels)
    elif representation == 'raster bilateral':
        return LearnableImageRasterBilateral(bilateral_blur, num_channels)
    elif representation == 'fourier':
        return LearnableImageFourier(height, width, num_channels)
    elif representation == 'raster':
        return LearnableImageRaster(height, width, num_channels)
    else:
        raise ValueError(f'Invalid method: {representation}')

# !!! 블렌딩 방식 변경해야함. ----------
def blend_torch_images(foreground, background, alpha):
    '주어진 foreground와 background 이미지를 alpha 값에 따라 블렌딩합니다.'
    assert foreground.shape == background.shape, 'Foreground와 background의 크기가 같아야 합니다.'
    C, H, W = foreground.shape
    assert alpha.shape == (H, W), 'alpha는 (H, W) 크기의 행렬이어야 합니다.'
    return foreground * alpha + background * (1 - alpha)

class PeekabooSegmenter(nn.Module):
    '이미지 분할을 위한 PeekabooSegmenter 클래스.'
    
    def __init__(self, 
                 audio: torch.Tensor, 
                 labels: List['BaseLabel'], 
                 mel_channel: int = 64,  # C
                 target_length: int = 1024,  # T
                 channel: int = 1, 
                 name: str = 'Untitled', 
                 bilateral_kwargs: dict = None, 
                 representation: str = 'fourier bilateral', 
                 min_step=None, 
                 max_step=None):
        super().__init__()     

        self.mel_channel, self.target_length = mel_channel, target_length  # (H,W)
        self.channel = channel
        self.labels = labels
        self.name = name
        self.representation = representation
        self.min_step = min_step
        self.max_step = max_step
        
        assert all(issubclass(type(label), BaseLabel) for label in labels), '모든 라벨은 BaseLabel의 서브클래스여야 합니다.'
        assert len(labels) > 0, '분할할 클래스가 최소 하나 이상 있어야 합니다.'
        
        # 오디오 마저 전처리: Torch (1,C,T) 형식 ((C,H,W) in image)
        self.foreground = audio
        assert [self.foreground.shape[i] for i in range(3)] == [self.channel, self.mel_channel, self.target_length], \
            f"오디오 멜-스펙트로그램의 크기나 값의 범위가 올바르지 않습니다. (현재 크기: {audio.shape})"
        # assert audio.min() >= 0 and audio.max() <= 1, '원소 min max 문제'
        print(audio.min(), audio.max())
        # assert audio.min() >= -0.5 and audio.max() <= 0.5, '원소 min max 문제'

        # 배경은 단색으로 설정 (0으로)
        self.background = torch.zeros_like(self.foreground)
        
        # 학습 가능한 알파 값 생성
        bilateral_kwargs = bilateral_kwargs or {}
        self.alphas = make_learnable_image(self.mel_channel, self.target_length, num_channels=len(labels), 
                                           foreground=self.foreground, 
                                           representation=self.representation, 
                                           bilateral_kwargs=bilateral_kwargs)  # !!! 알파 채널도 조정가능하게 수정해야함.
        # ---------- 그리고 일단 세로, 가로 변경해놓음 - 에러 안나는지 확인해보자
            
    @property
    def num_labels(self):
        '현재 레이블의 개수를 반환합니다.'
        return len(self.labels)
            
    def set_background_color(self, color):
        '배경 색상을 설정합니다. (각 채널 값은 0과 1 사이여야 함)'
        r,g,b = color
        assert 0 <= r <= 1 and 0 <= g <= 1 and 0 <= b <= 1, "각 채널 값은 0과 1 사이여야 합니다."
        self.background[0] = r
        # self.background[1] = g
        # self.background[2] = b
        
    def randomize_background(self):
        '배경 색상을 무작위로 설정합니다.'
        self.set_background_color(rp.random_rgb_float_color())
    
    def screening_background(self):
        '배경 색상을 지움으로 설정합니다.'
        self.set_background_color((0, 0, 0))
        
    def forward(self, alphas=None, return_alphas=False):
        '학습된 alpha 값을 이용해 이미지 분할을 수행하고 결과 이미지를 반환합니다.'
        
        # StableDiffusion 객체의 상태를 변경
        old_min_step, old_max_step = aldm.min_step, aldm.max_step
        aldm.min_step, aldm.max_step = self.min_step, self.max_step
        
        try:
            # alpha 값이 없으면, 학습된 alpha 생성
            alphas = alphas if alphas is not None else self.alphas()
            assert alphas.shape == (self.num_labels, self.mel_channel, self.target_length), "alphas 크기가 맞지 않습니다."
            assert alphas.min() >= 0 and alphas.max() <= 1, "alphas 값은 0과 1 사이여야 합니다."

            # alpha 값을 이용하여 각 라벨에 대한 이미지를 생성
            output_images = [blend_torch_images(self.foreground, self.background, alpha) for alpha in alphas]
            output_images = torch.stack(output_images)
            assert output_images.shape == (self.num_labels, self.channel, self.mel_channel, self.target_length), "출력 이미지 크기가 맞지 않습니다."

            return (output_images, alphas) if return_alphas else output_images

        finally:
            # StableDiffusion 객체의 원래 상태 복원
            aldm.min_step, aldm.max_step = old_min_step, old_max_step

def display(self):  # !!! 이거 배경색 입히지 말고 그냥 없애는 걸로 변경해야함 -----------.
    'PeekabooSegmenter의 이미지를 다양한 배경색과 함께 시각화하는 메서드.'

    # 기본 색상 설정 및 랜덤 색상 생성
    assert self.channel == 1 or 3, '채널이 1 또는 3이 아님.'
    if self.channel == 3:
        colors = [rp.random_rgb_float_color() for _ in range(3)]
    elif self.channel == 1:
        colors = [[0, 0, 0]]
    alphas = rp.as_numpy_array(self.alphas())
    assert alphas.shape == (self.num_labels, self.mel_channel, self.target_length)

    # 배경색과 함께 각 알파 채널로 생성된 이미지를 저장 -> 이미지는 '[[i1], [i2], [i3]]' 이런 형태
    composites = [rp.as_numpy_images(self(self.alphas())) for color in colors for _ in [self.set_background_color(color)]]

    # 레이블 이름 및 상태 정보 설정
    label_names = [label.name for label in self.labels]
    stats_lines = [self.name, '', f'H,W = {self.mel_channel}x{self.target_length}']

    # 전역 변수에서 특정 상태 정보를 추가
    for stat_format, var_name in [('Gravity: %.2e', 'GRAVITY'),
                                    ('Batch Size: %i', 'BATCH_SIZE'),
                                    ('Iter: %i', 'iter_num'),
                                    ('Image Name: %s', 'image_filename'),
                                    ('Learning Rate: %.2e', 'LEARNING_RATE'),
                                    ('Guidance: %i%%', 'GUIDANCE_SCALE')]:
        if var_name in globals():
            stats_lines.append(stat_format % globals()[var_name])

    # 이미지와 알파 채널을 각 배경색과 함께 결합하여 출력 이미지 생성
    output_image = rp.labeled_image(
        rp.tiled_images(
            rp.labeled_images(
                [self.image,
                    alphas[0],
                    composites[0][0],
                    composites[1][0] if len(composites) > 1 else None,
                    composites[2][0] if len(composites) > 2 else None],
                ["Input Image",
                    "Alpha Map",
                    "Background #1",
                    "Background #2" if len(composites) > 1 else None,
                    "Background #3" if len(composites) > 2 else None],),
            length=2 + len(composites),),
        label_names[0])

    # 이미지 출력
    rp.display_image(output_image)

    return output_image

PeekabooSegmenter.display=display

def get_mean_embedding(prompts:list):
    '주어진 프롬프트 리스트의 평균 임베딩을 계산하여 반환합니다'
    return torch.mean(
        torch.stack([aldm.get_text_embeddings(prompt) for prompt in prompts]),
        dim=0
    ).to(device)

class BaseLabel:
    '기본 레이블 클래스. 이름과 임베딩을 저장하며, 샘플 이미지를 생성할 수 있습니다'
    def __init__(self, name:str, embedding:torch.Tensor):
        #Later on we might have more sophisticated embeddings, such as averaging multiple prompts
        #We also might have associated colors for visualization, or relations between labels
        self.name=name
        self.embedding=embedding
        
    def get_sample_image(self):
        '임베딩을 기반으로 샘플 이미지를 생성하여 반환합니다.'
        output = aldm.embeddings_to_imgs(self.embedding)[0]
        assert rp.is_image(output), '생성된 출력이 이미지가 아닙니다.'
        return output

    def __repr__(self):
        return f"{type(self).__name__}(name={self.name})"
        
class SimpleLabel(BaseLabel):
    '텍스트 임베딩을 사용한 간단한 레이블 클래스.'
    def __init__(self, name:str):
        super().__init__(name, aldm.get_text_embeddings(name).to(device))

class MeanLabel(BaseLabel):
    '여러 프롬프트의 평균 임베딩을 사용한 레이블 클래스'
    #Test: rp.display_image(rp.horizontally_concatenated_images(MeanLabel('Dogcat','dog','cat').get_sample_image() for _ in range(1)))
    def __init__(self, name:str, *prompts):
        super().__init__(name, get_mean_embedding(rp.detuple(prompts)))

class PeekabooResults(EasyDict):
    'dict처럼 동작하지만 속성처럼 읽고 쓸 수 있는 클래스.'
    pass

def save_peekaboo_results(results,new_folder_path):
    'PeekabooResults를 지정된 폴더에 저장합니다.'

    import json
    assert not rp.folder_exists(new_folder_path), f'Please use a different name, not {new_folder_path}'
    rp.make_folder(new_folder_path)

    with rp.SetCurrentDirectoryTemporarily(new_folder_path):
        print(f"\nSaving PeekabooResults to {new_folder_path}")
        params = {}

        for key, value in results.items():
            if rp.is_image(value):
                rp.save_image(value, f'{key}.png')  # 단일 이미지 저장
            elif isinstance(value, np.ndarray) and rp.is_image(value[0]):
                rp.make_directory(key)  # 이미지 폴더 저장
                with rp.SetCurrentDirectoryTemporarily(key):
                    [rp.save_image(img, f'{i}.png') for i, img in enumerate(value)]
            elif isinstance(value, np.ndarray):
                np.save(f'{key}.npy', value)  # 일반 Numpy 배열 저장
            else:
                try:
                    json.dumps({key: value})  # JSON으로 변환 가능한 값 저장
                    params[key] = value
                except Exception:
                    params[key] = str(value)  # 변환 불가한 값은 문자열로 저장

        rp.save_json(params, 'params.json', pretty=True)
        print(f"Done saving PeekabooResults to {new_folder_path}!")

def run_peekaboo(name: str,
                 original_audio_file_path: Union[str, np.ndarray],
                 label: Optional['BaseLabel'] = None,

                 GRAVITY=1e-1/2,
                 NUM_ITER=300,
                 LEARNING_RATE=1e-5, 
                 BATCH_SIZE=1,   
                 GUIDANCE_SCALE=100,
                 bilateral_kwargs=dict(kernel_size=3, tolerance=0.08, sigma=5, iterations=40),

                 representation='fourier bilateral',
                 min_step=None, 
                 max_step=None,
                 
                 duration=10,
                 ) -> PeekabooResults:
    """
    Peekaboo Hyperparameters:
    GRAVITY=1e-1/2: prompt에 따라 tuning이 제일 필요함. 주로 1e-2, 1e-1/2, 1e-1, or 1.5*1e-1에서 잘 됨.
    NUM_ITER=300: 300이면 대부분 충분
    LEARNING_RATE=1e-5: neural neural textures 아닐 경우, 값 키워도 됨
    BATCH_SIZE=1: 큰 차이 없음. 배치 키우면 vram만 잡아먹음
    GUIDANCE_SCALE=100: DreamFusion 논문의 고정 값임.
    bilateral_kwargs=(kernel_size=3,tolerance=.08,sigma=5,iterations=40)
    square_image_method: input image를 정사각형화 하는 두 가지 방법. (crop / scale)
    representation: (fourier bilateral / raster bilateral / fourier / raster)
    """
    
    # 레이블이 없을 경우 기본 레이블 생성
    label = label or SimpleLabel(name)

    ## 이미지 로드 및 전처리
    # image_path = image if isinstance(image, str) else '<No image path given>'  # 지금 이미지가 array; (500 500 3) 이다.
    # image = rp.load_image(image_path) if isinstance(image, str) else image  # 내가 임의로 image_path로 바꾼거긴해..00
    # assert rp.is_image(image)
    # assert issubclass(type(label), BaseLabel)
    # image = rp.as_rgb_image(rp.as_float_image(make_image_square(image, square_image_method)))  # normalize하고, array; (512 512 3)으로 resize 되어있음

    # 오디오 로드 및 전처리
    audio_path = original_audio_file_path if isinstance(original_audio_file_path, str) else '<No audio path given>'
    
    audio_file_duration = ut.get_duration(original_audio_file_path)
    assert ut.get_bit_depth(original_audio_file_path) == 16, "The bit depth of the original audio file %s must be 16" % original_audio_file_path

    if(duration > audio_file_duration):
        print("Warning: Duration you specified %s-seconds must equal or smaller than the audio file duration %ss" % (duration, audio_file_duration))
        duration = ut.round_up_duration(audio_file_duration)
        print("Set new duration as %s-seconds" % duration)

    config = cf()
    fn_STFT = TacotronSTFT(
        config["preprocessing"]["stft"]["filter_length"],
        config["preprocessing"]["stft"]["hop_length"],
        config["preprocessing"]["stft"]["win_length"],
        config["preprocessing"]["mel"]["n_mel_channels"],
        config["preprocessing"]["audio"]["sampling_rate"],
        config["preprocessing"]["mel"]["mel_fmin"],
        config["preprocessing"]["mel"]["mel_fmax"],
        )
    
    mel, _, _ = wav_to_fbank(original_audio_file_path, target_length=int(duration * 102.4), fn_STFT=fn_STFT)  # (C,T)
    print(mel.shape)
    mel = mel.unsqueeze(0).to(device)  # (1,C,T)
    # mel = mel.unsqueeze(0)
    # mel = repeat(mel, "1 ... -> b ...", b=BATCH_SIZE)  # (batchsize, 1, C, T)    ----> 나중에 모델 입력 전에는 이런 꼴이어야 한다.

    # PeekabooSegmenter 생성
    pkboo = PeekabooSegmenter(audio=mel,
                            labels=[label],
                            name=name, 
                            bilateral_kwargs=bilateral_kwargs,
                            representation=representation,
                            min_step=min_step,
                            max_step=max_step
                            ).to(device)

    if 'bilateral' in representation:
        blur_image = rp.as_numpy_image(pkboo.alphas.bilateral_blur(pkboo.foreground))
        print("The bilateral blur applied to the input image before/after, to visualize it")
        # print(pkboo.foreground.shape)
        # reshaped_tensor = pkboo.foreground.squeeze(0).reshape(256, 256).unsqueeze(0)
        # expanded_tensor = reshaped_tensor.repeat(3, 1, 1).permute(1,2,0)
        # print(expanded_tensor.shape)
        # # 0과 256 사이로 값을 제한
        # tensor = (expanded_tensor - expanded_tensor.min()) / (expanded_tensor.max() - expanded_tensor.min()) * 1
        # np_array = tensor.cpu().numpy().astype(np.uint8)

        # rp.display_image(rp.tiled_images(rp.labeled_images([rp.as_numpy_image(np_array), blur_image], ['before', 'after'])))

    # pkboo.display()

    # 옵티마이저 설정
    params = list(pkboo.parameters())
    optim = torch.optim.SGD(params, lr=LEARNING_RATE)

    # 학습 반복 설정
    global iter_num
    iter_num = 0
    timelapse_frames=[]
    preview_interval = max(1, NUM_ITER // 10)  # 10번의 미리보기를 표시

    try:
        display_eta = rp.eta(NUM_ITER)
        for _ in range(NUM_ITER):
            display_eta(_)
            iter_num += 1

            alphas = pkboo.alphas()
            for __ in range(BATCH_SIZE):
                # pkboo.randomize_background()
                pkboo.screening_background()
                composites = pkboo()
                for label, composite in zip(pkboo.labels, composites):
                    aldm.train_step(label.embedding, composite[None], guidance_scale=GUIDANCE_SCALE)

            ((alphas.sum()) * GRAVITY).backward()
            optim.step()
            optim.zero_grad()

            # with torch.no_grad():
            #     if not _ % preview_interval: 
            #         timelapse_frames.append(pkboo.display())

    except KeyboardInterrupt:
        print("Interrupted early, returning current results...")
                

    results = PeekabooResults(
        #The main output
        alphas=rp.as_numpy_array(alphas),
        
        #Keep track of hyperparameters used
        GRAVITY=GRAVITY, BATCH_SIZE=BATCH_SIZE, NUM_ITER=NUM_ITER, GUIDANCE_SCALE=GUIDANCE_SCALE,
        bilateral_kwargs=bilateral_kwargs, representation=representation, label=label,
        audio=mel.permute(1, 2, 0), audio_path=audio_path, 
        
        #Record some extra info

        # preview_image=pkboo.display(), 
        # timelapse_frames=rp.as_numpy_array(timelapse_frames),
        **({'blur_image':blur_image} if 'blur_image' in dir() else {}),
        height=pkboo.mel_channel, width=pkboo.target_length, p_name=pkboo.name, min_step=pkboo.min_step, max_step=pkboo.max_step,
        
        device=device) 
    
    # 결과 폴더 생성 및 저장
    output_folder = rp.make_folder(f'peekaboo_results/{name}')
    output_folder += f'/{len(rp.get_subfolders(output_folder)):03}'
    save_peekaboo_results(results, output_folder)

    import matplotlib.pyplot as plt
    import librosa.display
    import numpy as np
    def plot_mel(mel, config, title="Mel Spectrogram"):
        """AudioLDM에서 사용하는 mel spectrogram을 직접 시각화합니다.
        
        Args:
            mel: shape (C, T) 또는 (1, C, T)의 mel spectrogram
            config: AudioLDM config
            title: 그래프 제목
        """
        plt.figure(figsize=(10, 10))
        
        # (1, C, T) -> (C, T) 변환
        if len(mel.shape) == 3:
            mel = mel.squeeze(0).permute(1, 0)
        
        # numpy로 변환
        if torch.is_tensor(mel):
            mel = mel.cpu().numpy()
        
        # 시각화
        librosa.display.specshow(
            mel,
            y_axis='mel',
            x_axis='time',
            sr=config["preprocessing"]["audio"]["sampling_rate"],
            hop_length=config["preprocessing"]["stft"]["hop_length"],
            cmap='viridis'
        )
        
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        plt.show()
    try:
        plot_mel(blur_image, config, title="Preprocessed Mel Spectrogram")

        plot_mel(pkboo(), config, title="Preprocessed Mel Spectrogram")
    except:
        pass


    return results
  
aldm=aldm.AudioLDM('cuda','cvssp/audioldm-s-full-v2')
device=aldm.device