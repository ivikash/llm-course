// viz/mel_spectrogram.js
// Shows audio (simulated sine waves) as a mel-spectrogram "image".

registerViz('mel_spectrogram', function (container) {
  container.innerHTML = `
    <p class="viz-title">Audio as a mel-spectrogram</p>
    <p class="viz-sub">Whisper turns audio into a 2D picture (freq × time), then runs a transformer on it.</p>
    <div class="viz-controls">
      <label>Audio type: <select id="ms-t">
        <option value="tone">Single tone</option>
        <option value="sweep" selected>Frequency sweep (chirp)</option>
        <option value="speech">Speech-like</option>
        <option value="music">Chord + melody</option>
      </select></label>
    </div>
    <canvas id="ms-canvas" width="600" height="300" style="image-rendering:pixelated;border:1px solid #d1d5db;border-radius:4px;max-width:100%"></canvas>
    <p class="viz-readout" id="ms-out">Horizontal axis = time. Vertical axis = frequency (low → high). Brightness = intensity.</p>
  `;
  const cvs = container.querySelector('#ms-canvas');
  const ctx = cvs.getContext('2d');
  function render() {
    const type = container.querySelector('#ms-t').value;
    const W = cvs.width, H = cvs.height;
    const img = ctx.createImageData(W, H);
    for (let t = 0; t < W; t++) {
      for (let f = 0; f < H; f++) {
        const freq = (H - f) / H;   // 0 (low) to 1 (high)
        const time = t / W;
        let energy = 0;
        if (type === 'tone') {
          energy = Math.abs(freq - 0.5) < 0.02 ? 1 : 0;
        } else if (type === 'sweep') {
          const expectedFreq = 0.1 + time * 0.8;
          energy = Math.max(0, 1 - Math.abs(freq - expectedFreq) * 15);
        } else if (type === 'speech') {
          const vowelCenter = 0.15 + 0.1 * Math.sin(time * 20);
          energy = Math.max(0, 1 - Math.abs(freq - vowelCenter) * 20);
          const harmonic = Math.max(0, 1 - Math.abs(freq - 2 * vowelCenter) * 30);
          energy += harmonic * 0.5;
          energy *= 0.3 + 0.7 * Math.abs(Math.sin(time * 8));
        } else {  // music
          const bassEnergy = Math.max(0, 1 - Math.abs(freq - 0.1) * 30);
          const midEnergy = Math.max(0, 1 - Math.abs(freq - 0.3) * 25);
          const trebleEnergy = Math.max(0, 1 - Math.abs(freq - (0.5 + 0.1 * Math.sin(time * 10))) * 25);
          energy = bassEnergy * 0.8 + midEnergy * 0.6 + trebleEnergy * (0.5 + 0.5 * Math.sin(time * 15));
        }
        energy = Math.min(1, energy);
        const i = (f * W + t) * 4;
        // viridis-like
        const r = Math.floor(68 + energy * 200);
        const g = Math.floor(1 + energy * 230);
        const b = Math.floor(84 - energy * 50);
        img.data[i] = r; img.data[i+1] = g; img.data[i+2] = b; img.data[i+3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }
  container.querySelector('#ms-t').addEventListener('change', render);
  render();
});
