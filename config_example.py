"""
Rename this file to config.py
and update the paths according to your system.
"""

# ==============================
# OASIS DATASET
# ==============================
OASIS_ROOT = r"PATH_TO_OASIS/OASIS_dis"
OASIS_CLINICAL = r"PATH_TO_OASIS/oasis_cross-sectional.xlsx"

# ==============================
# ADReSS DATASET
# ==============================
AUDIO_ROOT = r"PATH_TO_ADReSS/Full_wave_enhanced_audio"
TRANSCRIPT_ROOT = r"PATH_TO_ADReSS/transcription"

CC_META_PATH = r"PATH_TO_ADReSS/cc_meta_data.txt"
CD_META_PATH = r"PATH_TO_ADReSS/cd_meta_data.txt"

# ==============================
# SAVED EMBEDDINGS
# ==============================
EMB_FILE = "cnn_embeddings_raw.npy"
IDS_FILE = "cnn_mri_ids.npy"