import torch
import torchaudio
from nemo.collections.asr.models import EncDecRNNTBPEModel
from datetime import timedelta
import re
import os
import soundfile as sf
import numpy as np
import gc

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    td = timedelta(seconds=seconds)
    hours = int(td.total_seconds() // 3600)
    minutes = int((td.total_seconds() % 3600) // 60)
    secs = int(td.total_seconds() % 60)
    millis = int((td.total_seconds() % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def create_srt(segments, output_file):
    """Create SRT file from transcription segments"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{segment['start']} --> {segment['end']}\n")
            f.write(f"{segment['text']}\n\n")

def load_audio(audio_path, target_sr=16000):
    """Load audio file and convert to proper format"""
    try:
        waveform, sample_rate = sf.read(audio_path)
        waveform = torch.FloatTensor(waveform)
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.T
    except Exception as e:
        print(f"soundfile failed, trying torchaudio: {e}")
        waveform, sample_rate = torchaudio.load(audio_path)
    
    if sample_rate != target_sr:
        print(f"Resampling from {sample_rate}Hz to {target_sr}Hz...")
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
        sample_rate = target_sr
    
    if waveform.shape[0] > 1:
        print("Converting stereo to mono...")
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    return waveform, sample_rate

def chunk_audio(waveform, sample_rate, chunk_length_sec=20, overlap_sec=1):
    """Split audio into overlapping chunks"""
    chunk_samples = int(chunk_length_sec * sample_rate)
    overlap_samples = int(overlap_sec * sample_rate)
    step_samples = chunk_samples - overlap_samples
    
    total_samples = waveform.shape[1]
    chunks = []
    
    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunk = waveform[:, start:end]
        start_time = start / sample_rate
        chunks.append((chunk, start_time))
        
        if end >= total_samples:
            break
        start += step_samples
    
    print(f"Split audio into {len(chunks)} chunks of ~{chunk_length_sec}s each")
    return chunks

def save_temp_wav(waveform, sample_rate, temp_path="temp_audio.wav"):
    """Save waveform as temporary WAV file"""
    torchaudio.save(temp_path, waveform, sample_rate)
    return temp_path

def clear_memory():
    """Aggressively clear memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def transcribe_audio_to_srt(audio_path, output_srt, model_path="parakeet-tdt-0.6b-v3.nemo", 
                            chunk_length=20, overlap=1, language="en"):
    """
    Transcribe audio file to SRT with aggressive memory management
    Args:
        audio_path: Path to input audio file
        output_srt: Path to output SRT file
        model_path: Path to local .nemo model file
        chunk_length: Length of audio chunks in seconds (smaller = safer)
        overlap: Overlap between chunks in seconds
        language: Language code ('ru', 'en', etc.) - may not work for all models
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    print(f"Loading audio from {audio_path}...")
    waveform, sample_rate = load_audio(audio_path, target_sr=16000)
    
    duration = waveform.shape[1] / sample_rate
    print(f"Audio duration: {duration:.2f} seconds")
    
    # Split into chunks BEFORE loading model
    chunks = chunk_audio(waveform, sample_rate, chunk_length, overlap)
    
    # Free the full waveform
    del waveform
    clear_memory()
    
    print(f"\nLoading Parakeet TDT model from {model_path}...")
    model = EncDecRNNTBPEModel.restore_from(model_path)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Transcribe each chunk
    all_segments = []
    
    for idx, (chunk_waveform, start_time) in enumerate(chunks):
        print(f"\n{'='*50}")
        print(f"Processing chunk {idx + 1}/{len(chunks)} (at {start_time:.1f}s)")
        print(f"{'='*50}")
        
        temp_wav = f"temp_chunk_{idx}.wav"
        
        try:
            # Save chunk
            save_temp_wav(chunk_waveform, sample_rate, temp_wav)
            
            # Clear memory before processing
            clear_memory()
            
            with torch.no_grad():
                # Basic transcribe without language params (safer)
                # Language parameters may cause issues with some model versions
                result = model.transcribe([temp_wav], batch_size=1)[0]
                
                if hasattr(result, 'text'):
                    transcription = result.text
                elif isinstance(result, str):
                    transcription = result
                else:
                    transcription = str(result)
            
            # Clear memory immediately after transcription
            clear_memory()
            
            if not transcription or not transcription.strip():
                print(f"Warning: Empty transcription for chunk {idx + 1}")
                continue
            
            print(f"Preview: {transcription[:150]}...")
            
            # Split into sentences (simpler approach)
            words = transcription.split()
            max_words = 12
            sentences = []
            
            for i in range(0, len(words), max_words):
                sentence = ' '.join(words[i:i+max_words])
                if sentence.strip():
                    sentences.append(sentence)
            
            # Create segments with timing
            chunk_duration = chunk_waveform.shape[1] / sample_rate
            
            if sentences:
                segment_duration = chunk_duration / len(sentences)
                current_time = 0.0
                
                for sentence in sentences:
                    segment = {
                        'start': format_timestamp(start_time + current_time),
                        'end': format_timestamp(start_time + current_time + segment_duration),
                        'text': sentence.strip()
                    }
                    all_segments.append(segment)
                    current_time += segment_duration
            
        except Exception as e:
            print(f"ERROR processing chunk {idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except:
                    pass
            
            # Aggressive cleanup
            del chunk_waveform
            clear_memory()
    
    # Clean up model
    del model
    clear_memory()
    
    if not all_segments:
        raise Exception("No segments were transcribed successfully!")
    
    print(f"\n{'='*50}")
    print(f"Transcription complete: {len(all_segments)} segments")
    print(f"{'='*50}")
    
    # Create SRT file
    create_srt(all_segments, output_srt)
    print(f"✓ SRT file saved: {output_srt}")
    
    return all_segments

def batch_transcribe(audio_files, output_dir="output", model_path="parakeet-tdt-0.6b-v3.nemo"):
    """Transcribe multiple audio files to SRT"""
    os.makedirs(output_dir, exist_ok=True)
    
    for audio_file in audio_files:
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        output_srt = os.path.join(output_dir, f"{base_name}.srt")
        
        print(f"\n{'#'*60}")
        print(f"PROCESSING: {audio_file}")
        print(f"{'#'*60}")
        
        try:
            transcribe_audio_to_srt(audio_file, output_srt, model_path)
            clear_memory()  # Clear between files
        except Exception as e:
            print(f"ERROR processing {audio_file}: {e}")
            import traceback
            traceback.print_exc()
            clear_memory()

if __name__ == "__main__":
    audio_file = "input_audio.wav"
    srt_file = "output_subtitles.srt"
    model_file = "parakeet-tdt-0.6b-v3.nemo"
    
    try:
        # REDUCED chunk size for safety (smaller = less memory)
        segments = transcribe_audio_to_srt(
            audio_file, 
            srt_file, 
            model_file, 
            chunk_length=20,  # Reduced from 30 to 20 seconds
            overlap=1         # Reduced overlap
        )
        
        print(f"\n{'='*60}")
        print(f"SUCCESS! Created {len(segments)} subtitle segments")
        print(f"Output: {srt_file}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        clear_memory()
