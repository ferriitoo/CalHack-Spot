// function sendCommandOnChange() {
//     var commandInput = document.getElementById('command');
//     var command = commandInput.value;
//     var responseDiv = document.getElementById('response');

//     fetch('/send_command', {
//         method: 'POST',
//         body: new URLSearchParams({ 'command': command }),
//         headers: {
//             'Content-Type': 'application/x-www-form-urlencoded'
//         }
//     })
//     .then(response => response.text())
//     .then(data => {
//         responseDiv.innerText = 'Robot Response: ' + data;
//         commandInput.value = ''; // Clear the input field after submitting
//     })
//     .catch(error => {
//         responseDiv.innerText = 'Error: ' + error;
//         commandInput.value = ''; // Clear the input field after submitting, even in case of an error
//     });
// }


// let mediaRecorder;
// let audioChunks = [];

// const recordButton = document.getElementById("recordButton");
// const stopButton = document.getElementById("stopButton");
// const sendButton = document.getElementById("sendButton");
// const audioPlayer = document.getElementById("audioPlayer");

// recordButton.addEventListener("click", toggleRecording);
// stopButton.addEventListener("click", stopRecording);
// sendButton.addEventListener("click", sendAudioToBackend);

// async function toggleRecording() {
//     if (mediaRecorder && mediaRecorder.state === "recording") {
//         stopRecording();
//     } else {
//         startRecording();
//     }
// }

// function startRecording() {
//     const constraints = { audio: true };
//     navigator.mediaDevices.getUserMedia(constraints)
//         .then(function (stream) {
//             mediaRecorder = new MediaRecorder(stream);
//             mediaRecorder.ondataavailable = event => {
//                 audioChunks.push(event.data);
//             };
//             mediaRecorder.onstop = () => {
//                 const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
//                 audioPlayer.src = URL.createObjectURL(audioBlob);
//                 audioPlayer.style.display = "block";
//                 audioChunks = [];
//                 sendButton.disabled = false;
//             };
//             mediaRecorder.start();
//             recordButton.innerText = "Stop";
//             stopButton.disabled = false;
//         })
//         .catch(error => console.error('Error accessing the microphone:', error));
// }

// function stopRecording() {
//     if (mediaRecorder && mediaRecorder.state === "recording") {
//         mediaRecorder.stop();
//         recordButton.innerText = "Record";
//         stopButton.disabled = true;
//     }
// }
// function sendAudioToBackend() {
//     if (audioPlayer.src) {
//         const formData = new FormData();
//         formData.append('audio', audioBlob, 'audio.wav'); // Use the actual Blob instead of src

//         fetch('/save_audio', {
//             method: 'POST',
//             body: formData,
//         })
//         .then(response => {
//             if (response.status === 200) {
//                 console.log('Audio sent to backend successfully.');
//             } else {
//                 console.error('Error sending audio to backend.');
//             }
//         })
//         .catch(error => console.error('Error:', error));
//     }
// }


let mediaRecorder;
let audioChunks = [];

const recordButton = document.getElementById("recordButton");
const stopButton = document.getElementById("stopButton");
const sendButton = document.getElementById("sendButton");
const audioPlayer = document.getElementById("audioPlayer");

recordButton.addEventListener("click", toggleRecording);
stopButton.addEventListener("click", stopRecording);
sendButton.addEventListener("click", sendAudioToBackend);

async function toggleRecording() {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        stopRecording();
    } else {
        startRecording();
    }
}

function startRecording() {
    const constraints = { audio: true };
    navigator.mediaDevices.getUserMedia(constraints)
        .then(function (stream) {
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };
            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                audioPlayer.src = URL.createObjectURL(audioBlob);
                audioPlayer.style.display = "block";
                audioChunks = [];
                sendButton.disabled = false;
            };
            mediaRecorder.start();
            recordButton.innerText = "Stop";
            stopButton.disabled = false;
        })
        .catch(error => console.error('Error accessing the microphone:', error));
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        recordButton.innerText = "Record";
        stopButton.disabled = true;
    }
}

function sendAudioToBackend() {
    if (audioPlayer.src) {
        const formData = new FormData();
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' }); // Define audioBlob here
        formData.append('audio', audioBlob);

        fetch('/save_audio', {
            method: 'POST',
            body: formData,
        })
        .then(response => {
            if (response.status === 200) {
                console.log('Audio sent to backend successfully.');
            } else {
                console.error('Error sending audio to backend.');
            }
        })
        .catch(error => console.error('Error:', error));
    }
}
