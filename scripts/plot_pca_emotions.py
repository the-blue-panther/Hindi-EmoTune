import sys
sys.path.append('d:/Downloads/Bengali_EmoKnob')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

# Import necessary modules and functions from the notebook environment
# Note: This assumes XTTS model is already loaded and functions are available
from TTS.api import TTS

# Setup paths (from notebook)
PROJECT_ROOT = Path(r"D:\Downloads\Bengali_EmoKnob")
SR_XTTS = 22050
SR_INDIC = 16000

# Load XTTS model
print("Loading XTTS model...")
XTTS = TTS(model_name='tts_models/multilingual/multi-dataset/xtts_v2', gpu=False)
print("XTTS loaded.")

# Import librosa and processor for embeddings
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Load Indic model
INDIC_LOCAL = Path('models') / 'ai4bharat_indicwav2vec_hindi'
print('Loading Indic wav2vec from:', INDIC_LOCAL)
processor = Wav2Vec2Processor.from_pretrained(str(INDIC_LOCAL))
indic_enc = Wav2Vec2Model.from_pretrained(str(INDIC_LOCAL))
indic_enc.eval()
print('Indic encoder loaded.')

# Helper functions from notebook
def resolve_xtts_internal_model(tts_obj):
    '''Return the internal XTTS model used by TTS wrapper.'''
    if tts_obj is None:
        raise RuntimeError('Provided tts_obj is None')
    if hasattr(tts_obj, 'synthesizer') and hasattr(tts_obj.synthesizer, 'tts_model'):
        return tts_obj.synthesizer.tts_model
    if hasattr(tts_obj, 'tts_model'):
        return tts_obj.tts_model
    raise RuntimeError('Could not resolve internal XTTS model. Ensure you loaded native XTTS-v2 via TTS API.')

def get_xtts_speaker_latent(tts_obj, wav_path, load_sr=SR_XTTS):
    '''Extract GPT conditioning latent (prosody) from XTTS.
    Returns flattened numpy array (approx 32k dims).'''
    model = resolve_xtts_internal_model(tts_obj)
    try:
        res = model.get_conditioning_latents(str(wav_path), load_sr=load_sr)
    except TypeError:
        res = model.get_conditioning_latents(str(wav_path))
        
    if isinstance(res, (list, tuple)) and len(res) >= 2:
        # res[0] is gpt_cond_latent (1, 32, 1024) -> flatten to (32768,)
        # res[1] is speaker_embedding (1, 512, 1)
        gpt_lat = res[0]
    else:
        # Fallback if structure is different
        gpt_lat = res
        
    try:
        sp = gpt_lat.cpu().numpy() if hasattr(gpt_lat, 'cpu') else np.array(gpt_lat)
        return sp.ravel() # Flatten to 1D (~32768)
    except Exception as e:
        raise RuntimeError('Failed to convert GPT latent to numpy: ' + str(e))

def plot_emotion_samples_pca(emotion_name, output_filename=None):
    """Plot emotion samples in 2D PCA space showing neutral vs emotion shifts.
    
    Args:
        emotion_name: name of emotion folder (e.g., 'happy', 'sad', 'angry')
        output_filename: optional filename to save the plot
    """
    emotion_dir = PROJECT_ROOT / 'data' / 'emotion_samples' / emotion_name
    
    if not emotion_dir.exists():
        print(f'Emotion folder not found: {emotion_dir}')
        return
    
    neutral_vecs = []
    emotion_vecs = []
    sample_names = []
    
    sample_dirs = sorted([d for d in emotion_dir.iterdir() if d.is_dir()])
    
    print(f'🎵 Loading {emotion_name} samples...')
    
    for sd in sample_dirs:
        n_clean = sd / 'neutral_clean.wav'
        e_clean = sd / f'{emotion_name}_clean.wav'
        
        # Skip if clean files don't exist
        if not (n_clean.exists() and e_clean.exists()):
            print(f'  ⚠️ {sd.name}: missing clean files, skipping')
            continue
        
        try:
            # Extract speaker latents
            n_emb = get_xtts_speaker_latent(XTTS, n_clean, load_sr=SR_XTTS)
            e_emb = get_xtts_speaker_latent(XTTS, e_clean, load_sr=SR_XTTS)
            
            neutral_vecs.append(n_emb)
            emotion_vecs.append(e_emb)
            sample_names.append(sd.name)
            print(f'  ✓ {sd.name}')
        except Exception as e:
            print(f'  ✗ {sd.name}: {str(e)[:40]}')
    
    if len(neutral_vecs) == 0:
        print(f'No valid samples found in {emotion_dir}')
        return
    
    neutral_vecs = np.array(neutral_vecs)  # shape: (n_samples, 32768)
    emotion_vecs = np.array(emotion_vecs)  # shape: (n_samples, 32768)
    
    # Compute averages
    neutral_avg = neutral_vecs.mean(axis=0)  # (32768,)
    emotion_avg = emotion_vecs.mean(axis=0)  # (32768,)
    
    # Stack all vectors for PCA
    all_vecs = np.vstack([neutral_vecs, emotion_vecs, neutral_avg.reshape(1, -1), emotion_avg.reshape(1, -1)])
    
    # Apply PCA to 2D
    pca = PCA(n_components=2)
    vecs_2d = pca.fit_transform(all_vecs)
    
    # Split back
    n_samples = len(neutral_vecs)
    neutral_2d = vecs_2d[:n_samples]
    emotion_2d = vecs_2d[n_samples:2*n_samples]
    neutral_avg_2d = vecs_2d[2*n_samples]
    emotion_avg_2d = vecs_2d[2*n_samples + 1]
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Plot individual samples
    plt.scatter(neutral_2d[:, 0], neutral_2d[:, 1], c='blue', s=100, alpha=0.6, 
               label='Neutral samples', edgecolors='darkblue', linewidth=1.5)
    plt.scatter(emotion_2d[:, 0], emotion_2d[:, 1], c='red', s=100, alpha=0.6, 
               label=f'{emotion_name.capitalize()} samples', edgecolors='darkred', linewidth=1.5)
    
    # Annotate sample names
    for i, name in enumerate(sample_names):
        plt.annotate(name, (neutral_2d[i, 0], neutral_2d[i, 1]), 
                    fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')
        plt.annotate(name, (emotion_2d[i, 0], emotion_2d[i, 1]), 
                    fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')
    
    # Plot averages (larger markers)
    plt.scatter(neutral_avg_2d[0], neutral_avg_2d[1], c='blue', s=400, marker='X', 
               edgecolors='darkblue', linewidth=2, label='Neutral avg', zorder=10)
    plt.scatter(emotion_avg_2d[0], emotion_avg_2d[1], c='red', s=400, marker='X', 
               edgecolors='darkred', linewidth=2, label=f'{emotion_name.capitalize()} avg', zorder=10)
    
    # Draw arrow from neutral to emotion average (emotion shift)
    plt.arrow(neutral_avg_2d[0], neutral_avg_2d[1], 
             emotion_avg_2d[0] - neutral_avg_2d[0], 
             emotion_avg_2d[1] - neutral_avg_2d[1],
             head_width=0.2, head_length=0.15, fc='green', ec='green', alpha=0.7, linewidth=2.5, zorder=5)
    
    # Labels and formatting
    explained_var = pca.explained_variance_ratio_
    cumsum_var = np.cumsum(explained_var)
    
    plt.xlabel(f'PC1 ({explained_var[0]:.1%})', fontsize=12)
    plt.ylabel(f'PC2 ({explained_var[1]:.1%})', fontsize=12)
    plt.title(f'Emotion Vectors: {emotion_name.capitalize()}\nPCA 2D Projection (Cumulative: {cumsum_var[1]:.1%})',
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure if filename provided
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f'✓ Saved PCA plot to {output_filename}')
    
    plt.show()
    
    # Print statistics
    print(f'\n📊 PCA Statistics:')
    print(f'   PC1 variance: {explained_var[0]:.2%}')
    print(f'   PC2 variance: {explained_var[1]:.2%}')
    print(f'   Cumulative: {cumsum_var[1]:.2%}')
    
    print(f'\n📈 Vector Statistics ({n_samples} samples):')
    print(f'   Neutral avg norm: {np.linalg.norm(neutral_avg):.3f}')
    print(f'   {emotion_name.capitalize()} avg norm: {np.linalg.norm(emotion_avg):.3f}')
    emotion_diff = emotion_avg - neutral_avg
    print(f'   Difference norm: {np.linalg.norm(emotion_diff):.3f}')
    cosine_sim = np.dot(neutral_avg, emotion_avg) / (np.linalg.norm(neutral_avg) * np.linalg.norm(emotion_avg) + 1e-12)
    print(f'   Cosine similarity: {cosine_sim:.3f}')


# Generate plots for all 3 emotions
if __name__ == "__main__":
    print("Generating PCA plots for all emotions...\n")
    
    emotions = ['angry', 'happy', 'sad']
    
    for emotion in emotions:
        print(f"\n{'='*70}")
        print(f"Processing: {emotion.upper()}")
        print(f"{'='*70}")
        output_file = f'{emotion}_pca.png'
        plot_emotion_samples_pca(emotion, output_filename=output_file)
        print()
    
    print("\n" + "="*70)
    print("✓ All PCA plots generated successfully!")
    print("="*70)
