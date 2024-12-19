# Torch and model dependencies
import torch
import torchaudio
from torchaudio.transforms import Resample, MFCC
from AudioClassifier import AudioClassifier, N_MFCC, TARGET_SR, CLASS_MAP, DURATION

# Fast API things
from fastapi import FastAPI, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Other imports
from io import BytesIO


# Model configuration
model = AudioClassifier()
model.load_state_dict(torch.load("model.pth", weights_only=True, map_location=torch.device('cpu')))
model.eval()

# Define the FastAPI app
app = FastAPI()

# Allow CORS from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# Audio preprocessing
def preprocess_audio(data, target_sr=TARGET_SR):
    waveform, sr = data

    # Resample if necessary
    if sr != target_sr:
        resample = Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resample(waveform)

    # Duration in samples
    num_samples = target_sr * DURATION

    # Trim or pad waveform
    if waveform.shape[1] > num_samples:
        waveform = waveform[:, :num_samples]
    else:
        padding = num_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))

    # Extract MFCCs
    mfcc = MFCC(
        sample_rate=target_sr,
        n_mfcc=N_MFCC,
        melkwargs={"n_fft": 1024, "hop_length": 512, "n_mels": 64}
    )(waveform)

    # Ensure single channel dimension
    mfcc = mfcc.unsqueeze(1)

    return mfcc

# Prediction API
@app.post("/predict_instrument")
async def predict_instrument(file: bytes = File(...)):
    try:
        # Load bytes from the payload
        data = torchaudio.load(BytesIO(file))
        
        # Preprocess audio
        mfcc = preprocess_audio(data)
        
        # Predict
        with torch.no_grad():
            output = model(mfcc)
            _, predicted_class = torch.max(output, 1)
        # map to class
        class_name = CLASS_MAP[predicted_class.tolist()[0]]
        return {"predicted_class": class_name}
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))


# Start the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)