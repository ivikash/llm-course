// viz/whisper_streaming.js
// Audio -> mel chunks -> streaming transcription.

registerViz('whisper_streaming', function (container) {
  container.innerHTML = `
    <p class="viz-title">Whisper: streaming speech-to-text</p>
    <p class="viz-sub">Audio arrives in 30s chunks. Each chunk → log-mel spectrogram → encoder → decoder generates text tokens.</p>
    <div class="viz-controls">
      <button id="ws-play">▶ Transcribe live</button>
      <button id="ws-reset">Reset</button>
    </div>
    <div style="margin-top:12px">
      <div style="font-size:11px;color:#6b7280;margin-bottom:4px">Audio waveform (simulated)</div>
      <canvas id="ws-wave" width="700" height="80" style="background:#111;border-radius:4px;width:100%;max-width:700px"></canvas>
    </div>
    <div style="margin-top:12px;display:flex;gap:12px;flex-wrap:wrap">
      <div style="flex:1;min-width:200px">
        <div style="font-size:11px;color:#6b7280;margin-bottom:4px">Current chunk → mel spectrogram</div>
        <canvas id="ws-mel" width="200" height="80" style="background:#fff;border:1px solid #d1d5db;border-radius:4px;image-rendering:pixelated"></canvas>
      </div>
      <div style="flex:2;min-width:260px">
        <div style="font-size:11px;color:#6b7280;margin-bottom:4px">Generated transcript</div>
        <div id="ws-text" style="background:#fff;border:1px solid #d1d5db;border-radius:4px;padding:10px;min-height:70px;font-family:ui-monospace,monospace;font-size:13px;line-height:1.5"></div>
      </div>
    </div>
    <p class="viz-readout" id="ws-out"></p>
  `;

  const fullText = "Hello, this is a demonstration of Whisper's streaming transcription. Audio arrives in thirty-second chunks.".split(' ');
  const waveCvs = container.querySelector('#ws-wave');
  const waveCtx = waveCvs.getContext('2d');
  const melCvs = container.querySelector('#ws-mel');
  const melCtx = melCvs.getContext('2d');
  let playing = false;

  function drawWave(pos) {
    const W = waveCvs.width, H = waveCvs.height;
    waveCtx.fillStyle = '#111';
    waveCtx.fillRect(0, 0, W, H);
    waveCtx.strokeStyle = '#10b981';
    waveCtx.lineWidth = 1;
    waveCtx.beginPath();
    for (let x = 0; x < W; x++) {
      const amp = (x < pos ? 0.6 : 0.15) * (Math.sin(x * 0.15) + Math.sin(x * 0.3 + 1) * 0.7 + Math.random() * 0.3);
      waveCtx.moveTo(x, H/2);
      waveCtx.lineTo(x, H/2 + amp * H * 0.4);
      waveCtx.moveTo(x, H/2);
      waveCtx.lineTo(x, H/2 - amp * H * 0.4);
    }
    waveCtx.stroke();
    // playhead
    waveCtx.strokeStyle = '#f59e0b';
    waveCtx.lineWidth = 2;
    waveCtx.beginPath();
    waveCtx.moveTo(pos, 0); waveCtx.lineTo(pos, H);
    waveCtx.stroke();
  }

  function drawMel() {
    const W = 200, H = 80;
    const cols = 40, rows = 20;
    const cw = W / cols, ch = H / rows;
    for (let c = 0; c < cols; c++) {
      for (let r = 0; r < rows; r++) {
        const v = Math.abs(Math.sin(c * 0.4 + r * 0.2 + Math.random())) * (1 - r / rows * 0.5);
        const intensity = Math.floor(v * 255);
        melCtx.fillStyle = `hsl(${240 - v * 180}, 80%, ${30 + v * 40}%)`;
        melCtx.fillRect(c * cw, (rows - 1 - r) * ch, cw, ch);
      }
    }
  }

  async function play() {
    if (playing) return;
    playing = true;
    container.querySelector('#ws-text').innerHTML = '';
    const W = waveCvs.width;
    const wordDur = W / fullText.length;
    for (let i = 0; i < fullText.length; i++) {
      if (!playing) break;
      drawWave((i + 1) * wordDur);
      if (i % 4 === 0) drawMel();
      const span = document.createElement('span');
      span.textContent = fullText[i] + ' ';
      span.style.opacity = '0';
      container.querySelector('#ws-text').appendChild(span);
      requestAnimationFrame(() => { span.style.transition = 'opacity 0.2s'; span.style.opacity = '1'; });
      const chunkBoundary = (i + 1) % 8 === 0;
      container.querySelector('#ws-out').innerHTML =
        `Processed ${Math.ceil((i+1)*30/fullText.length)} seconds of audio. ` +
        (chunkBoundary ? '<b>New 30s chunk → new mel spectrogram → encoder/decoder run.</b>' : 'Streaming decode continues...');
      await new Promise(r => setTimeout(r, 180));
    }
    playing = false;
  }

  drawWave(0);
  drawMel();
  container.querySelector('#ws-play').addEventListener('click', play);
  container.querySelector('#ws-reset').addEventListener('click', () => {
    playing = false;
    container.querySelector('#ws-text').innerHTML = '';
    drawWave(0);
  });
});
