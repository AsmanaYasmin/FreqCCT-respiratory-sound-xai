# FreqCCT: Multi-Representation Compact Convolutional Transformer for Respiratory Sound Classification with Explainable AI

1. Overview
Respiratory diseases—including Chronic Obstructive Pulmonary Disease (COPD), asthma, and lower respiratory infections—are major global causes of morbidity and mortality. Traditional chest auscultation is subjective and depends heavily on physician experience and clinical environment acoustics. Automated auscultation systems based on deep learning can enable objective, accessible pulmonary screening.
This repository presents FreqCCT, a lightweight, multi-representation Compact Convolutional Transformer architecture for automated respiratory sound classification using the ICBHI 2017 Benchmark and KAUH Respiratory Database (1,667 audio recordings across 7 disease classes). The framework stacks multi-frequency acoustic representations (Mel-spectrogram, Constant-Q Transform, and MFCC) and couples the transformer backbone with extensive Explainable AI (XAI) validation—including Grad-CAM, Attention Rollout, Integrated Gradients, and Exact Channel Shapley values—to provide clinically interpretable digital biomarker attributions.
2. Processing Pipeline & Model Architecture
• Acoustic Preprocessing:  Audio signals are resampled to 4 kHz and divided into 5-second frames with 50% overlap. Baseline-wander removal (0–1 Hz DFT bins zeroed) and max-abs normalization ensure uniform waveform quality.
• Class Balancing Augmentation:  Waveform augmentation on non-COPD classes (time stretch 0.85–1.15, pitch shift ±3 semitones, Gaussian noise 0.001–0.01) balances each class to 9,845 frames.
• 3-Channel Time-Frequency Stacking:  For each frame, three complementary time-frequency maps are extracted: Mel-spectrogram (64 bands, 64 × 39), Constant-Q Transform (CQT, 64 bins from 32.7 Hz, 64 × 40), and MFCC (30 coefficients, 30 × 40). Features are Min-Max scaled to [0, 1], resized to 64 × 40, and stacked into a 3-channel tensor (3 × 64 × 40).
• Dual-Stage Tokenization:  Two convolutional tokenizers (Tokenizer 1: 3×3 conv, 3→64, ReLU, max-pool; Tokenizer 2: depthwise 3×3 + pointwise 1×1, 64→64, ReLU, max-pool) project inputs into 160 tokens × 64 dimensions (16 frequency rows × 10 time columns) with learned positional embeddings.
• Transformer Backbone & Head:  4 Pre-LN Transformer encoder blocks (4 attention heads, d = 64, MLP expansion 64→128→64 with GELU, residual connections). Classification head performs LayerNorm, temporal pooling over 10 time tokens, keeping 16 frequency rows (1,024 features), and linear projection to 7 disease classes at both frame and recording levels.
```text
[Respiratory Audio (ICBHI 2017 & KAUH - 1,667 Recordings, 7 Classes)]
                                │
                                ▼
[Preprocessing: 4 kHz Resampling, 5 s Frames (50% Overlap), Baseline Removal]
                                │
                                ▼
[Multi-Frequency Feature Extraction: Mel (64) + CQT (64) + MFCC (30)]
                                │
                                ▼
[Min-Max Scaling & 3-Channel Stacking (3 × 64 × 40 Tensor)]
                                │
                                ▼
[Dual-Stage Conv Tokenizers → 160 Tokens × 64 + Positional Embedding]
                                │
                                ▼
[4-Block Pre-LN Transformer Encoder (4 Heads, d = 64, GELU MLP)]
                                │
                                ▼
[Classification Head: Frame-Level (7 Classes) & Recording-Level Argmax]
                                │
                                ▼
[Multi-Tier XAI: Grad-CAM, Attention Rollout, Integrated Gradients, Shapley]
3. Explainability (XAI) Suite & Cost Analysis
A comprehensive multi-method interpretability protocol validates that FreqCCT bases predictions on legitimate acoustic pathology rather than background artifacts:
XAI Method / Test	Protocol & Configuration	Clinical & Interpretability Objective
Grad-CAM Saliency	True class attribution on 16 × 10 token grid	Identifies frequency-temporal energy concentrations.
Attention Rollout	Attention weights combination (0.5 w + 0.5 t)	Visualizes information flow across self-attention layers.
Integrated Gradients	Training-mean baseline across 64 steps	Validates path-integrated feature attributions.
Rollout vs Grad-CAM Agreement	Spearman rank correlation, top-10% IoU, shuffle null	Confirms consistency across gradient and attention maps.
Exact Channel Shapley Values	Mel, CQT, MFCC across 8 coalitions	Quantifies independent diagnostic contribution per channel.

• ICBHI Pointing Test:  Validates attention alignment against expert annotations for adventitious lung sounds (crackle and wheeze cycles).
• Computational Complexity:  Evaluated per 5-second frame—covering total parameters, MFLOPs (self-attention included), INT8 quantized model weight size, peak activation memory, and single-thread CPU runtime for edge healthcare deployment.
4. Dataset Description & Partitioning
The framework is benchmarked on multi-center respiratory sound databases:
• Datasets:  1,667 respiratory sound recordings across 7 diagnostic categories (COPD, Healthy, Asthma, Bronchiectasis, Bronchiolitis, Pneumonia, URTI).
• Evaluation Scheme:  Frame-level stratified 5-fold cross-validation (after class-balancing augmentation). Test set comprises real unaugmented frames only (n = 2,360 in fold 0), with 10% validation split and an additional patient-disjoint fold check to prevent data leakage.
5. How to Run (Execution Guide Matching Repository)
The repository structure corresponds directly to your uploaded files on GitHub:
• `full_process_clean.ipynb`: Master Google Colab / Jupyter Notebook implementing preprocessing, feature extraction, FreqCCT training, and XAI evaluations.
• `methodology_diagram_v3.pdf` & `methodology_diagram_v2.png`: High-resolution end-to-end architecture diagrams (Processing pipeline, Pre-LN Encoder block, and Explanation/Cost analysis).
• `architecture OF cct.pdf`: Structural schematic and layer-wise specifications of the Compact Convolutional Transformer.
• `README.md`: Repository overview, architecture flowchart, evaluation benchmarks, and run instructions.
Step 1: Environment Setup & Dependencies
Clone the repository and install required audio processing, deep learning, and explainability packages:
# Clone the repository
git clone https://github.com/AsmanaYasmin/FreqCCT-respiratory-sound-xai.git
cd FreqCCT-respiratory-sound-xai

# Install dependencies
pip install torch torchvision torchaudio librosa scipy numpy pandas matplotlib seaborn captum

Step 2: Executing via Google Colab / JupyterLab
Open `full_process_clean.ipynb` in Google Colab or JupyterLab:
1. Mount Google Drive and set the path to the raw ICBHI 2017 audio and annotation files.
2. Execute the notebook cells sequentially:
   • Section 1: Preprocessing (4 kHz resampling, baseline wander removal, 5 s segmentation with 50% overlap).
   • Section 2: Audio Augmentation (time stretch, pitch shifting, Gaussian noise injection for non-COPD classes).
   • Section 3: Time-Frequency Feature Extraction (Mel-spectrogram, CQT, and MFCC generation, Min-Max scaling, and 3x64x40 tensor stacking).
   • Section 4: FreqCCT Model Training (Convolutional tokenizers, Pre-LN Transformer encoder, 5-fold cross-validation with AdamW optimizer).
   • Section 5: Frame-Level & Recording-Level Evaluation (Stratified real-test frame validation and confusion matrices).
   • Section 6: Explainable AI Suite (Generates Grad-CAM token heatmaps, Attention Rollout maps, Integrated Gradients, and Channel Shapley attributions).
Step 3: Verification & Output Reproducibility
The notebook exports explanation maps on the 16 × 10 token grid, ICBHI pointing test metrics, and computational latency benchmarks (MFLOPs and CPU time per 5-second frame).
