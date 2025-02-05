import os
import sys

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(proj_dir, 'src_audioldm')
sys.path.extend([proj_dir, src_dir])

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
import rp
import soundfile as sf
import src_audioldm.audioldm as ldm
from src_audioldm.utilities.data.dataset_pkb import AudioDataProcessor, spectral_normalize_torch

from src_audioldm.learnable_textures2 import AudioMaskGeneratorCNN, AudioMaskGeneratorTransformer

ldm = ldm.AudioLDM('cuda:0')
device = ldm.device


def iterative_audio_transform(ldm, audioprocessor, initial_audio, target_text, transfer_strength, num_iterations=5):
    current_audio = initial_audio  # 첫 번째 입력 오디오
    for i in range(2, num_iterations + 2):  # att2.wav ~ att{num_iterations+1}.wav
        output_audio = f"./att{i}.wav"
        
        # AudioLDM을 사용하여 스타일 변환 수행
        waveform = ldm.style_transfer(
            text=target_text,
            original_audio_file_path=current_audio,
            transfer_strength=transfer_strength,
            processor=audioprocessor,
        )

        # 변환된 오디오 저장
        waveform = waveform.squeeze(0).detach().cpu().numpy()
        sf.write(output_audio, waveform, 16000)

        print(f"Generated: {output_audio}")
        
        # 다음 단계의 입력 오디오를 현재 출력으로 설정
        current_audio = output_audio


def run_peekaboo(target_text: str,
                 audio_file_path: str,
                 GRAVITY=1e-1/2,      # prompt에 따라 tuning이 제일 필요. (1e-2, 1e-1/2, 1e-1, 1.5*1e-1)
                 NUM_ITER=300,        # 이정도면 충분
                 LEARNING_RATE=1e-5,  # neural neural texture 아니면 키워도 됨.
                 BATCH_SIZE=1,        # 키우면 vram만 잡아먹음
                 GUIDANCE_SCALE=100,  # DreamFusion 참고하여 default값 설정
                 representation='fourier bilateral',
                 min_step=None,
                 max_step=None):

    audioprocessor = AudioDataProcessor(device=device)
    dataset = audioprocessor.preprocessing_data(audio_file_path)
    dataset2 = audioprocessor.preprocessing_data("./best_samples/Footsteps_on_a_wooden_floor.wav")

    datasep1, datasep2 = audioprocessor.get_mixed_batches(dataset, dataset2)

    '''
    {
    "text":         # list
    "fname":        # list
    "waveform":     # tensor, [B, 1, samples_num]
    "stft":         # tensor, [B, t-steps, f-bins]
    "log_mel_spec": # tensor, [B, t-steps, mel-bins]
    }
    '''
    assert len(dataset['text']) == 1, 'Only one audio file is allowed'
    # assert dataset['text'][0] == target_text, 'Text does not match'
    if dataset['text'][0] != target_text:
        dataset['text'][0] = target_text
        print("Warning!: Text has been changed to match the target_text")

    # waveform = ldm.style_transfer(
    #     text=target_text,
    #     # original_audio_file_path="/home/sba2024/MIIL_ZSS/audio_peekaboo/peekaboo_results/A cat meowing/011/seped_audio.wav",
    #     original_audio_file_path="./att2.wav",
    #     transfer_strength=0.4,
    #     processor=audioprocessor,
    #     )
    
    # waveform = waveform.squeeze(0).detach().numpy()
    # sf.write("./att3.wav", waveform, 16000)


    # 예제 실행
    initial_audio_path = "/home/sba2024/MIIL_ZSS/audio_peekaboo/peekaboo_results/A cat meowing/011/seped_audio.wav"
    target_text = "A cat meowing"
    transfer_strength = 0.4
    num_iterations = 5  # 반복 횟수 (att2.wav ~ att6.wav 생성)

    iterative_audio_transform(ldm, audioprocessor, initial_audio_path, target_text, transfer_strength, num_iterations)

    raise ValueError

    def train_step():
        alphas = pkboo.alphas()
        composite_set = pkboo()
        
        dummy_for_plot = ldm.train_step(composite_set, guidance_scale=GUIDANCE_SCALE)
        
        loss = alphas.mean() * GRAVITY
        alphaloss = loss.item()
        # loss2 = torch.abs(alphas[:, 1:, :] - alphas[:, :-1, :]).mean() + torch.abs(alphas[:, :, 1:] - alphas[:, :, :-1]).mean()
        # loss += loss2 * 5000
        # print(loss2.item())
        loss.backward(); optim.step(); optim.zero_grad()
        sdsloss, uncond, cond, eps_diff = dummy_for_plot
        return sdsloss, alphaloss, uncond, cond, eps_diff

    list_sds, list_alpha, list_uncond_eps, list_cond_eps, list_eps_differ = [], [], [], [], []
    list_dummy = (list_sds, list_alpha, list_uncond_eps, list_cond_eps, list_eps_differ)
    try:
        for iter_num in tqdm(range(NUM_ITER)):
            dummy_for_plot = train_step()
            for li, element in zip(list_dummy, dummy_for_plot):
                li.append(element)

    except KeyboardInterrupt:
        print("Interrupted early, returning current results...")
        pass
    
    alphas = pkboo.alphas()

    results = {
        "alphas":rp.as_numpy_array(alphas),
        
        "representation":representation,
        "NUM_ITER":NUM_ITER,
        "GRAVITY":GRAVITY,
        "lr":LEARNING_RATE,
        "GUIDANCE_SCALE":GUIDANCE_SCALE,
        "BATCH_SIZE":BATCH_SIZE,
        
        "target_text":pkboo.text[0],
        "device":device,
    }
    
    output_folder = rp.make_folder('peekaboo_results/%s'%target_text)
    output_folder += '/%03i'%len(rp.get_subfolders(output_folder))
    save_peekaboo_results(results, output_folder, list_dummy)

    mel_cpu = dataset['log_mel_spec'][0, ...].detach().cpu()
    save_melspec_as_img(mel_cpu, os.path.join(output_folder, "GT_mel.png"))

    alpha = make_learnable_image(513, 1024, 1, representation).to(device)
    mel_cpu = pkboo(alpha())['log_mel_spec'][0, ...].detach().cpu()
    save_melspec_as_img(mel_cpu, os.path.join(output_folder, "mixed_mel.png"))

    mel_cpu = pkboo()['log_mel_spec'][0, ...].detach().cpu()
    save_melspec_as_img(mel_cpu, os.path.join(output_folder, "seped_mel.png"))

    import soundfile as sf
    mel = dataset['log_mel_spec'][0, ...].half()
    audio = ldm.post_process_from_mel(mel)
    sf.write(os.path.join(output_folder, "GT_audio.wav"), audio, 16000)

    mel = pkboo()['log_mel_spec'][0, ...].half()
    audio = ldm.post_process_from_mel(mel)
    sf.write(os.path.join(output_folder, "seped_audio.wav"), audio, 16000)
    
    print(f">> Saved results at {output_folder}!!!")

def save_melspec_as_img(mel_tensor, save_path):
    mel = mel_tensor.detach().cpu().numpy()
    if mel.shape[0] > mel.shape[1]:
        mel = mel.T  # (64, 1024)로 전치
    height, width = mel.shape
    aspect_ratio = width / height  # 1024/64 = 16
    fig_width = 20  # 기준 가로 길이
    fig_height = fig_width / aspect_ratio  # 20/16 = 1.25
    if mel.min() < 0:
        # min_, max_ = -11.5129, 3.4657
        min_, max_ = mel.min(), mel.max()
    else:
        min_, max_ = 0, 1
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(mel, aspect='auto', origin='lower', cmap='magma',
            vmin=min_, vmax=max_)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def save_peekaboo_results(results, new_folder_path, list_dummy):
    import json
    assert not rp.folder_exists(new_folder_path), f'Please use a different name, not {new_folder_path}'
    rp.make_folder(new_folder_path)
    with rp.SetCurrentDirectoryTemporarily(new_folder_path):
        print(f"Saving PeekabooResults to {new_folder_path}")
        params = {}
        for key, value in results.items():
            if rp.is_image(value):  # Save a single image
                rp.save_image(value, f'{key}.png')
            elif isinstance(value, np.ndarray) and rp.is_image(value[0]):  # Save a folder of images
                rp.make_directory(key)
                with rp.SetCurrentDirectoryTemporarily(key):
                    for i in range(len(value)):
                        rp.save_image(value[i], f'{i}.png')
            elif isinstance(value, np.ndarray):  # Save a generic numpy array
                np.save(f'{key}.npy', value) 
            else:
                try:
                    json.dumps({key: value})
                    params[key] = value  #Assume value is json-parseable
                except Exception:
                    params[key] = str(value)
        rp.save_json(params, 'params.json', pretty=True)
        print(f"Done saving PeekabooResults to {new_folder_path}!")
    
    # Loss plot 저장
    sds, alpha, uncond, cond, eps = list_dummy
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 1, 1); plt.plot(sds, label='SDS Loss')
    plt.xlabel('Iteration'); plt.ylabel('Loss'); plt.legend()
    plt.subplot(2, 1, 2); plt.plot(alpha, label='Alpha Loss')
    plt.xlabel('Iteration'); plt.ylabel('Loss'); plt.legend()
    plt.tight_layout(); plt.savefig(f'{new_folder_path}/loss_plot.png'); plt.close()

    plt.figure(figsize=(25, 10))
    plt.subplot(3, 1, 1); plt.plot(uncond, label='uncond')
    plt.xlabel('Iteration'); plt.ylabel('abs mean'); plt.legend()
    plt.subplot(3, 1, 2); plt.plot(cond, label='cond')
    plt.xlabel('Iteration'); plt.ylabel('abs mean'); plt.legend()
    plt.subplot(3, 1, 3); plt.plot(eps, label='difference bet eps')
    plt.xlabel('Iteration'); plt.ylabel('abs mean'); plt.legend()
    plt.tight_layout(); plt.savefig(f'{new_folder_path}/eps_plot.png'); plt.close()

if __name__ == "__main__":
    
    # # bilateral fourier 용도.
    # prms = {
    #     'G': 3000,
    #     'iter': 300,
    #     'lr': 1e-5,
    #     'B': 1,
    #     'guidance': 100,
    #     'representation': 'fourier bilateral',
    # }

    # raster 용도.
    prms = {
        'G': 5000, # 3000,
        'iter': 250,
        'lr': 0.00001,
        'B': 1,
        'guidance': 100,
        'representation': 'cnn',
    }

    # run_peekaboo(
    #     target_text='Footsteps on a wooden floor', # 'A cat meowing',
    #     audio_file_path="./best_samples/A_cat_meowing.wav",
    #     GRAVITY=prms['G'],
    #     NUM_ITER=prms['iter'],
    #     LEARNING_RATE=prms['lr'],
    #     BATCH_SIZE=prms['B'],
    #     GUIDANCE_SCALE=prms['guidance'],
    #     representation=prms['representation'],
    #     )
    
    run_peekaboo(
        target_text='A cat meowing', # 'A cat meowing',
        audio_file_path="./best_samples/A_cat_meowing.wav",
        GRAVITY=prms['G'],
        NUM_ITER=prms['iter'],
        LEARNING_RATE=prms['lr'],
        BATCH_SIZE=prms['B'],
        GUIDANCE_SCALE=prms['guidance'],
        representation=prms['representation'],
        )
