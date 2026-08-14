// Inject HUD UI on Google Meet
const hud = document.createElement('div');
hud.id = 'fakeshield-hud';
hud.innerHTML = `
  <div class="fs-title">🛡️ FakeShield Live</div>
  <div id="fs-status">Monitoring stream...</div>
  <div class="fs-meter">
    <span>Deepfake Risk:</span>
    <span id="fs-score" style="font-weight:bold;">0%</span>
  </div>
`;
document.body.appendChild(hud);

const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');
let isProcessing = false;

function playAlertSound() {
  try {
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(800, audioCtx.currentTime);
    gain.gain.setValueAtTime(0.1, audioCtx.currentTime);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.2);
  } catch (e) {}
}

async function processLiveCall() {
  if (isProcessing) return;

  const videoElements = Array.from(document.querySelectorAll('video')).filter(
    (v) => v.readyState === 4 && v.videoWidth > 0
  );

  if (videoElements.length > 0) {
    const video = videoElements[0];
    canvas.width = 224;
    canvas.height = 224;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const base64Frame = canvas.toDataURL('image/jpeg', 0.65);

    isProcessing = true;
    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: base64Frame }),
      });

      const data = await response.json();

      if (data.status === 'success') {
        const scorePercent = Math.round(data.fake_score * 100);
        const statusEl = document.getElementById('fs-status');
        const scoreEl = document.getElementById('fs-score');

        scoreEl.innerText = `${scorePercent}%`;

        if (data.is_fake) {
          hud.className = 'alert';
          statusEl.innerText = '⚠️ DEEPFAKE DETECTED';
          scoreEl.style.color = '#f87171';
          playAlertSound();
        } else {
          hud.className = '';
          statusEl.innerText = '✅ Authentic Feed';
          scoreEl.style.color = '#4ade80';
        }
      }
    } catch (err) {
      document.getElementById('fs-status').innerText = '⚠️ Backend offline';
    } finally {
      isProcessing = false;
    }
  }

  setTimeout(processLiveCall, 100);
}

setTimeout(processLiveCall, 3000);
