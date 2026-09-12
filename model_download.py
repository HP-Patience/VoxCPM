from modelscope import snapshot_download
snapshot_download("OpenBMB/VoxCPM2", local_dir='./pretrained_models/VoxCPM2') # specify the local directory to save the model

from voxcpm import VoxCPM
import soundfile as sf
model = VoxCPM.from_pretrained("./pretrained_models/VoxCPM2", load_denoiser=False)

wav = model.generate(
    text="VoxCPM2 is the current recommended release for realistic multilingual speech synthesis.",
    cfg_value=2.0,
    inference_timesteps=10,
)
sf.write("demo.wav", wav, model.tts_model.sample_rate)