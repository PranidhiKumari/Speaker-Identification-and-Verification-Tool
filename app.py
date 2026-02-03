import os
import numpy as np
import streamlit as st
import sounddevice as sd
import torch
import librosa
from sklearn.metrics.pairwise import cosine_similarity
from speechbrain.pretrained import EncoderClassifier

# ======================= CONFIG =======================
SAMPLE_RATE = 16000
RECORD_SECONDS = 5
DB_PATH = "speakers_db"
MIN_ENROLLMENTS = 3
DEFAULT_THRESHOLD = 0.75

os.makedirs(DB_PATH, exist_ok=True)


# ======================= LOAD MODEL =======================

# ======================= LOAD MODEL (LOCAL) =======================
@st.cache_resource
def get_model():
    from speechbrain.pretrained import EncoderClassifier
    model = EncoderClassifier.from_hparams(
        source="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"}
    )
    model.eval()
    return model

model = get_model()



# ======================= AUDIO RECORD =======================
def record_audio():
    st.info("🎙️ Recording... Speak clearly")
    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    return audio.flatten()

# ======================= AUDIO PROCESS =======================
def preprocess_audio(audio):
    audio, _ = librosa.effects.trim(audio, top_db=25)

    if len(audio) < SAMPLE_RATE * 2:
        return None, "Audio too short (min 2 sec)"

    if np.mean(audio ** 2) < 0.0005:
        return None, "Audio too silent"

    audio = audio / (np.max(np.abs(audio)) + 1e-9)
    return torch.tensor(audio).unsqueeze(0), None

# ======================= EMBEDDING =======================
def extract_embedding(signal):
    with torch.no_grad():
        emb = model.encode_batch(signal)
    emb = emb.squeeze().cpu().numpy()

    # SAFETY CHECK
    if emb.size == 0:
        return None
    return emb

# ======================= DATABASE HELPERS =======================
def speaker_files(name):
    return (
        os.path.join(DB_PATH, name + ".npy"),
        os.path.join(DB_PATH, name + "_count.txt")
    )

def get_enroll_count(name):
    _, count_path = speaker_files(name)
    if not os.path.exists(count_path):
        return 0
    return int(open(count_path).read())

def save_speaker(name, new_emb):
    emb_path, count_path = speaker_files(name)

    # 🚨 SAFETY: ignore broken old files
    if os.path.exists(emb_path):
        try:
            old_emb = np.load(emb_path)
            if old_emb.size != new_emb.size:
                raise ValueError("Corrupt embedding")
            new_emb = np.mean([old_emb, new_emb], axis=0)
        except Exception:
            # Delete broken file
            os.remove(emb_path)
            new_emb = new_emb

    count = get_enroll_count(name) + 1

    np.save(emb_path, new_emb)
    with open(count_path, "w") as f:
        f.write(str(count))

    return count

def load_db():
    db = {}
    for file in os.listdir(DB_PATH):
        if file.endswith(".npy"):
            name = file.replace(".npy", "")
            if get_enroll_count(name) >= MIN_ENROLLMENTS:
                try:
                    db[name] = np.load(os.path.join(DB_PATH, file))
                except:
                    pass
    return db

# ======================= STREAMLIT UI =======================
st.title("🎤 Speaker Identification (Stable Version)")

mode = st.sidebar.radio("Mode", ["➕ Add Speaker", "🔍 Identify Speaker"])

# ======================= ADD SPEAKER =======================
if mode == "➕ Add Speaker":
    speaker_name = st.text_input("Speaker Name")

    if st.button("Record & Enroll"):
        audio = record_audio()
        signal, err = preprocess_audio(audio)

        if err:
            st.error(err)
        else:
            emb = extract_embedding(signal)
            if emb is None:
                st.error("Embedding failed, try again")
            else:
                count = save_speaker(speaker_name, emb)
                st.success(f"Enrollment {count}/{MIN_ENROLLMENTS}")

# ======================= IDENTIFY =======================
if mode == "🔍 Identify Speaker":
    THRESHOLD = st.slider("Threshold", 0.65, 0.85, DEFAULT_THRESHOLD, 0.01)

    if st.button("Record & Identify"):
        audio = record_audio()
        signal, err = preprocess_audio(audio)

        if err:
            st.error(err)
        else:
            test_emb = extract_embedding(signal)
            db = load_db()

            if not db:
                st.warning("No fully enrolled speakers")
            else:
                scores = {
                    name: cosine_similarity(
                        test_emb.reshape(1, -1),
                        ref.reshape(1, -1)
                    )[0][0]
                    for name, ref in db.items()
                }

                best = max(scores, key=scores.get)
                score = scores[best]

                st.json(scores)

                if score >= THRESHOLD:
                    st.success(f"KNOWN: {best} ({score:.2f})")
                else:
                    st.error(f"UNKNOWN (closest {best}: {score:.2f})")