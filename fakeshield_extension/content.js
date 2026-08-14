// FakeShield - Google Meet Deepfake Detection Extension

const API_URL = "https://virtual-eye.onrender.com/predict";


// ===============================
// CREATE FAKE SHIELD HUD
// ===============================

const hud = document.createElement("div");

hud.id = "fakeshield-hud";

hud.innerHTML = `
    <div class="fs-title">
        🛡️ FakeShield Live
    </div>

    <div id="fs-status">
        Monitoring stream...
    </div>

    <div class="fs-meter">
        <span>Deepfake Risk:</span>
        <span id="fs-score" style="font-weight:bold;">
            0%
        </span>
    </div>
`;

document.body.appendChild(hud);


// ===============================
// CANVAS
// ===============================

const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d");

let isProcessing = false;
let lastFakeState = false;


// ===============================
// ALERT SOUND
// ===============================

function playAlertSound() {

    try {

        const audioCtx =
            new (window.AudioContext ||
                window.webkitAudioContext)();

        const oscillator =
            audioCtx.createOscillator();

        const gain =
            audioCtx.createGain();

        oscillator.type = "sawtooth";

        oscillator.frequency.setValueAtTime(
            800,
            audioCtx.currentTime
        );

        gain.gain.setValueAtTime(
            0.1,
            audioCtx.currentTime
        );

        oscillator.connect(gain);
        gain.connect(audioCtx.destination);

        oscillator.start();

        oscillator.stop(
            audioCtx.currentTime + 0.2
        );

    } catch (error) {

        console.log(
            "Audio alert unavailable:",
            error
        );
    }
}


// ===============================
// FIND VIDEO
// ===============================

function getVideoElement() {

    const videos =
        Array.from(
            document.querySelectorAll("video")
        );

    const activeVideos =
        videos.filter(
            video =>
                video.readyState >= 2 &&
                video.videoWidth > 0 &&
                video.videoHeight > 0
        );

    if (activeVideos.length === 0) {
        return null;
    }

    // Choose the largest visible video
    activeVideos.sort(
        (a, b) =>
            (b.videoWidth * b.videoHeight) -
            (a.videoWidth * a.videoHeight)
    );

    return activeVideos[0];
}


// ===============================
// PROCESS LIVE CALL
// ===============================

async function processLiveCall() {

    if (isProcessing) {
        return;
    }


    const video =
        getVideoElement();


    // No active video yet
    if (!video) {

        const statusEl =
            document.getElementById(
                "fs-status"
            );

        if (statusEl) {
            statusEl.innerText =
                "Waiting for video...";
        }

        setTimeout(
            processLiveCall,
            1500
        );

        return;
    }


    // ===============================
    // CAPTURE FRAME
    // ===============================

    canvas.width = 224;
    canvas.height = 224;

    ctx.drawImage(
        video,
        0,
        0,
        224,
        224
    );


    const base64Frame =
        canvas.toDataURL(
            "image/jpeg",
            0.65
        );


    isProcessing = true;


    try {

        const response =
            await fetch(
                API_URL,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        image: base64Frame
                    })
                }
            );


        // Check HTTP status first
        if (!response.ok) {

            throw new Error(
                `Server returned HTTP ${response.status}`
            );
        }


        // Read response as text first
        // This prevents JSON.parse errors
        const responseText =
            await response.text();


        let data;

        try {

            data =
                JSON.parse(
                    responseText
                );

        } catch (jsonError) {

            console.error(
                "Server did not return JSON:",
                responseText
            );

            throw new Error(
                "Backend returned invalid JSON"
            );
        }


        // ===============================
        // UPDATE HUD
        // ===============================

        const statusEl =
            document.getElementById(
                "fs-status"
            );

        const scoreEl =
            document.getElementById(
                "fs-score"
            );


        if (!statusEl || !scoreEl) {
            return;
        }


        // ===============================
        // HANDLE YOUR BACKEND RESPONSE
        // ===============================

        if (data.status === "success") {

            const fakeScore =
                Number(
                    data.fake_score || 0
                );

            const scorePercent =
                Math.round(
                    fakeScore * 100
                );


            scoreEl.innerText =
                `${scorePercent}%`;


            if (data.is_fake) {

                hud.className =
                    "alert";

                statusEl.innerText =
                    "⚠️ DEEPFAKE DETECTED";

                scoreEl.style.color =
                    "#f87171";


                // Play sound only when state changes
                if (!lastFakeState) {

                    playAlertSound();

                }

                lastFakeState = true;


            } else {

                hud.className = "";

                statusEl.innerText =
                    "✅ Authentic Feed";

                scoreEl.style.color =
                    "#4ade80";

                lastFakeState = false;
            }


        } else {

            statusEl.innerText =
                data.message ||
                "Analysis unavailable";

            scoreEl.innerText =
                "--";
        }


    } catch (error) {

        console.error(
            "FakeShield API error:",
            error
        );


        const statusEl =
            document.getElementById(
                "fs-status"
            );

        const scoreEl =
            document.getElementById(
                "fs-score"
            );


        if (statusEl) {

            statusEl.innerText =
                "⚠️ Backend connection error";
        }


        if (scoreEl) {

            scoreEl.innerText =
                "--";
        }


    } finally {

        isProcessing = false;

        // Analyze approximately every 1.5 seconds
        setTimeout(
            processLiveCall,
            1500
        );
    }
}


// ===============================
// START
// ===============================


setTimeout(
    processLiveCall,
    3000
);