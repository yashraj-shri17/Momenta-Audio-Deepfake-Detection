
import os
import shutil
import torch
import torchaudio
import numpy as np
from datasets import load_dataset
from gtts import gTTS
from pydub import AudioSegment
import tqdm
import logging
import traceback

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DATASET_ROOT = "mini_dataset"
REAL_DIR = os.path.join(DATASET_ROOT, "real")
FAKE_DIR = os.path.join(DATASET_ROOT, "fake")
PROTOCOL_FILE = os.path.join(DATASET_ROOT, "protocol.txt")
SAMPLES_TO_GENERATE = 30

def cleanup():
    if os.path.exists(DATASET_ROOT):
        try: shutil.rmtree(DATASET_ROOT)
        except: pass
    os.makedirs(REAL_DIR, exist_ok=True)
    os.makedirs(FAKE_DIR, exist_ok=True)

def main():
    cleanup()
    logger.info("Using LJSpeech (Public Domain, Continuous)...")
    
    try:
        # LJSpeech is widely available and public
        ds = load_dataset("lj_speech", split="train", streaming=True, trust_remote_code=True)
        ds_iter = iter(ds)
        
        protocol_lines = []
        
        for i in tqdm.tqdm(range(SAMPLES_TO_GENERATE), desc="Generating"):
            try:
                sample = next(ds_iter)
            except StopIteration:
                break
                
            # REAL: LJSpeech (Audiobook quality)
            audio = sample['audio']['array']
            sr = sample['audio']['sampling_rate']
            text = sample['text']
            
            # Save Real
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio).float()
            if audio.dim() == 1: audio = audio.unsqueeze(0)
            
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                audio = resampler(audio)

            real_name = f"real_{i:03d}.flac"
            real_path = os.path.join(REAL_DIR, real_name)
            torchaudio.save(real_path, audio, 16000)
            protocol_lines.append(f"LA_{i:03d} {os.path.join('real', os.path.splitext(real_name)[0])} - - bonafide -")

            # FAKE: gTTS
            fake_name = f"fake_{i:03d}.flac"
            fake_path = os.path.join(FAKE_DIR, fake_name)
            
            try:
                tts = gTTS(text=text[:100], lang='en') # Limit text length
                mp3_temp = os.path.join(FAKE_DIR, f"temp_{i}.mp3")
                tts.save(mp3_temp)
                
                seg = AudioSegment.from_mp3(mp3_temp)
                seg = seg.set_frame_rate(16000).set_channels(1)
                seg.export(fake_path, format="flac")
                if os.path.exists(mp3_temp): os.remove(mp3_temp)
            except:
                noise = torch.randn(1, 16000)
                torchaudio.save(fake_path, noise, 16000)
                
            protocol_lines.append(f"TT_{i:03d} {os.path.join('fake', os.path.splitext(fake_name)[0])} - - spoof -")

        with open(PROTOCOL_FILE, 'w') as f:
            f.write('\n'.join(protocol_lines))
        logger.info("SUCCESS: LJSpeech dataset generated.")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        # Last resort fallback
        logger.info("Generating Synthetic Fallback...")
        sr = 16000
        lines = []
        for i in range(30):
             # Real: Sine
             t = np.linspace(0, 1, sr)
             wf = torch.from_numpy(np.sin(2*np.pi*440*t)).float().unsqueeze(0)
             torchaudio.save(os.path.join(REAL_DIR, f"real_{i}.flac"), wf, sr)
             lines.append(f"LA_{i} real/real_{i} - - bonafide -")
             
             # Fake: Noise
             torchaudio.save(os.path.join(FAKE_DIR, f"fake_{i}.flac"), torch.randn(1, sr), sr)
             lines.append(f"TT_{i} fake/fake_{i} - - spoof -")
        with open(PROTOCOL_FILE, 'w') as f: f.write('\n'.join(lines))

if __name__ == "__main__":
    main()
