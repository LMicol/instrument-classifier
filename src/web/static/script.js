let mediaRecorder;
let isRecording = false;
let currentStream = null;
const statusElement = document.getElementById("status");
const startMicrophoneButton = document.getElementById("start-microphone");
let audioChunks = [];

// Validate DOM elements
if (!statusElement || !startMicrophoneButton) {
    console.error("Required DOM elements not found");
    throw new Error("Required DOM elements not found");
}

function highlightSquare(prediction) {
    const instruments = ["drums", "piano", "guitar", "violin", "mute"];
    instruments.forEach(instrument => {
        const element = document.getElementById(instrument);
        if (element) {
            if (instrument === prediction) {
                element.classList.add("shine");
            } else {
                element.classList.remove("shine");
            }
        }
    });
}

async function toggleRecording() {
    if (isRecording) {
        mediaRecorder.stop();
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
        }
        statusElement.textContent = "Stopped microphone capture.";
        isRecording = false;
        startMicrophoneButton.textContent = "Start Microphone Capture";
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        currentStream = stream;
        mediaRecorder = new MediaRecorder(stream);
        isRecording = true;
        startMicrophoneButton.textContent = "Stop Microphone Capture";
        statusElement.textContent = "Listening...";

        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
            if (audioChunks.length > 0) {
                sendAudioToAPI(audioChunks);
                audioChunks = [];
            }
            if (isRecording) {
                setTimeout(() => {
                    if (isRecording) {
                        mediaRecorder.start();
                        setTimeout(() => mediaRecorder.stop(), 2000);
                    }
                }, 1000);
            }
        };

        mediaRecorder.start();
        setTimeout(() => mediaRecorder.stop(), 2000);
    } catch (error) {
        statusElement.textContent = `Error accessing microphone: ${error.message}`;
        isRecording = false;
        startMicrophoneButton.textContent = "Start Microphone Capture";
    }
}

async function sendAudioToAPI(audioChunks) {
    const chunkBlob = new Blob(audioChunks, { 
        type: mediaRecorder.mimeType 
    });

    const formData = new FormData();
    formData.append('file', chunkBlob, 'audio.webm');

    try {
        const response = await fetch('http://localhost:8000/predict_instrument', {
            method: 'POST',
            body: formData,
        });

        if (response.ok) {
            const result = await response.json();
            console.log(result);
            highlightSquare(result.predicted_class);
        } else {
            statusElement.textContent = `Error: ${response.statusText}`;
        }
    } catch (error) {
        statusElement.textContent = `Error: ${error.message}`;
        audioChunks = []; // Clear chunks on error
    }
}

startMicrophoneButton.addEventListener("click", toggleRecording);