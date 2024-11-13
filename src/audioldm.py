from typing import Union, List, Optional

from diffusers import AudioLDMPipeline, PNDMScheduler
from transformers import ClapTextModelWithProjection, RobertaTokenizer, RobertaTokenizerFast, SpeechT5HifiGan
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import default_audioldm_config
import rp


class AudioLDM(nn.Module):
    def __init__(self, device='cuda', checkpoint_path="cvssp/audioldm-s-full"):
        super().__init__()
        self.device = torch.device(device)
        self.num_train_timesteps = 1000

        # Timestep ~ U(0.02, 0.98) to avoid very high / low noise levels
        self.min_step = int(self.num_train_timesteps * 0.02)  # aka 20
        self.max_step = int(self.num_train_timesteps * 0.98)  # aka 980

        pipe = AudioLDMPipeline.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float16, ##
            safety_checker=None,
            # variant="fp16",
            # low_cpu_mem_usage=True  <-- Vram 부족하면 설정
        ).to(self.device)

        pipe.scheduler = PNDMScheduler(beta_start=0.00085,
                                            beta_end=0.012,
                                            beta_schedule="scaled_linear",
                                            num_train_timesteps=self.num_train_timesteps) # Error from scheduling_lms_discrete.py

        self.pipe         = pipe
        self.vae          = pipe.vae.to(self.device)           # AutoencoderKL
        self.tokenizer    = pipe.tokenizer                     # Union[RobertaTokenizer, RobertaTokenizerFast]
        self.text_encoder = pipe.text_encoder.to(self.device)  # ClapTextModelWithProjection
        self.unet         = pipe.unet.to(self.device)          # UNet2DConditionModel
        self.vocoder      = pipe.vocoder.to(self.device)       # SpeechT5HifiGan
        self.scheduler    = pipe.scheduler                     # PNDMScheduler

        assert isinstance(self.vae, AutoencoderKL), type(self.vae)
        assert isinstance(self.tokenizer, Union[RobertaTokenizer, RobertaTokenizerFast]), type(self.tokenizer)
        assert isinstance(self.text_encoder, ClapTextModelWithProjection), type(self.text_encoder)
        assert isinstance(self.unet, UNet2DConditionModel), type(self.unet)
        assert isinstance(self.vocoder, SpeechT5HifiGan), type(self.vocoder)
        assert isinstance(self.scheduler, PNDMScheduler), type(self.scheduler)

        self.checkpoint_path = checkpoint_path
        self.uncond_text = ''
        '''
        - 빈 문자열 (''): 
            텍스트 조건이 전혀 필요하지 않은 상황에서는 빈 문자열을 사용하여 임베딩이 최소한의 영향을 받도록 할 수 있습니다.
        - 일반적인 텍스트 (' ', 'None' 등): 
            오디오 도메인에서 특정 조건이 없음을 의미하는 텍스트를 설정할 수 있습니다.
            예를 들어, 음성 생성 모델의 경우,
            'silence', 'background noise', 'ambient'와 같은 텍스트를 조건으로 추가해볼 수 있습니다.
        '''
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience
        print(f'[INFO] Loaded AudioLDM model from {checkpoint_path}')

        self.config = default_audioldm_config()

    def get_text_embeddings(self, prompts: Union[str, List[str]]) -> torch.Tensor:
        # if isinstance(prompts, str):
        #     prompts = [prompts]
        
        # print(prompts)
        # # 토큰화 & 임베딩 추출
        # text_input = self.tokenizer(prompts,
        #                             padding='max_length',
        #                             max_length=self.tokenizer.model_max_length,
        #                             truncation=True,
        #                             return_tensors='pt'
        #                             ).input_ids
        # print(text_input.shape)
        # with torch.no_grad():
        #     text_embeddings = self.text_encoder(text_input.to(self.device))[0]
        # print(text_embeddings.shape)

        
        # # unconditional embeddings에 대해서도 동일 수행
        # uncond_input = self.tokenizer([self.uncond_text] * len(prompts),
        #                               padding='max_length',
        #                               max_length=self.tokenizer.model_max_length,
        #                               return_tensors='pt'
        #                               ).input_ids
        # with torch.no_grad():
        #     uncond_embeddings = self.text_encoder(uncond_input.to(self.device))[0]

        # # 검증
        # assert len(uncond_embeddings) == len(text_embeddings) == len(prompts), \
        #     f"Length mismatch: {len(uncond_embeddings)}, {len(text_embeddings)}, {len(prompts)}"
        
        # # unconditional embedding 일관성 검사
        # assert (uncond_embeddings == uncond_embeddings[0].unsqueeze(0)).all(), \
        #     "Unconditional embeddings are not consistent"

        # # 결합 및 shape 확인
        # output_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # assert (uncond_embeddings == torch.stack([uncond_embeddings[0]] * len(uncond_embeddings))).all()
        # # assert output_embeddings.shape == (len(prompts) * 2, 77, 768), f"Embedding shape: {text_embeddings.shape}"
        # print(text_embeddings.shape)
        # return output_embeddings
            # `prompts`가 문자열이면 리스트로 변환
        if isinstance(prompts, str):
            prompts = [prompts]
        
        print(prompts)  # 입력 prompt 확인
        
        # prompt와 negative prompt 설정
        prompt = prompts
        negative_prompt = [self.uncond_text] * len(prompts)
        
        # `_encode_prompt` 호출
        output_embeddings = self.pipe._encode_prompt(
            prompt=prompt,
            device=self.device,
            num_waveforms_per_prompt=1,  # 필요한 경우 설정 변경 가능
            do_classifier_free_guidance=True,  # guidance 여부 설정
            negative_prompt=negative_prompt,
        )
        
        # `output_embeddings` shape 확인
        print(output_embeddings.shape)
        
        return output_embeddings

    def train_step(self,
                   text_embeddings: torch.Tensor,
                   pred_audio: torch.Tensor,        # (11CT) 리소스 문제 때문에 미리 줄였다 원복해서 썼어야 했을까 생각해보기
                   guidance_scale: float = 100,
                   t: Optional[int] = None):
        pred_audio = pred_audio.to(torch.float16)

        if t is None:
            t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        assert 0 <= t < self.num_train_timesteps, 'invalid timestep t=%i' % t

        # encode image into latents with vae, requires grad
        latents = self.encode_audios(pred_audio)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t.cpu())
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=None, class_labels=text_embeddings).sample

        # perform guidance (high scale from paper!) ----- ###
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        grad = w * (noise_pred - noise)

        # manually backward, since we omitted an item in grad and cannot simply autodiff
        latents.backward(gradient=grad, retain_graph=True)
        return 0  # dummy loss value



########## 여기 채우는 거는 기존 원본 stable diffusion 코드랑 ldm 전처리과정을 참고하자.

    def encode_audios(self, audios: torch.Tensor) -> torch.Tensor:

        assert len(audios.shape) == 4 and audios.shape[1] == 1  # [B, 1, C, T]

        audios = 2 * audios - 1
        posterior = self.vae.encode(audios)

        # 모델에서 std 기준으로 scaling 해주는 걸로 알고 있긴 한데,, ###
        if hasattr(posterior, 'latent_dist'):
            latents = posterior.latent_dist.sample() * 0.18215
        else:
            latents = posterior.sample() * 0.18215

        assert len(latents.shape) == 4 and latents.shape[1] == 8  # [B, 4, C, T]
        assert latents.shape[1:] == (8,16,256), f'{latents.shape[1:]}' ##
        return latents



