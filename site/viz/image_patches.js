// viz/image_patches.js
// Shows how ViT turns an image into patches → tokens.
// Uses a canvas with a simple generated image (or user upload).

registerViz('image_patches', function (container) {
  container.innerHTML = `
    <p class="viz-title">Image → patches → tokens (ViT)</p>
    <p class="viz-sub">ViT splits an image into a grid of patches. Each patch → one linear embedding → one 'token'. Same transformer processes them as if they were words.</p>
    <div class="viz-controls">
      <label>Patch size: <input id="ip-p" type="range" min="8" max="64" step="4" value="32"><span id="ip-p-out">32px</span></label>
      <label>Image: <select id="ip-img">
        <option value="gradient">Gradient</option>
        <option value="circles" selected>Circles</option>
        <option value="stripes">Stripes</option>
      </select></label>
    </div>
    <div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center">
      <canvas id="ip-orig" width="256" height="256" style="border:1px solid #d1d5db;border-radius:4px"></canvas>
      <canvas id="ip-patched" width="256" height="256" style="border:1px solid #d1d5db;border-radius:4px"></canvas>
    </div>
    <p class="viz-readout" id="ip-out"></p>
  `;
  const orig = container.querySelector('#ip-orig');
  const patched = container.querySelector('#ip-patched');
  const o = orig.getContext('2d');
  const p = patched.getContext('2d');

  function drawImage(which) {
    o.clearRect(0, 0, 256, 256);
    if (which === 'gradient') {
      const g = o.createLinearGradient(0, 0, 256, 256);
      g.addColorStop(0, '#3b82f6'); g.addColorStop(1, '#f59e0b');
      o.fillStyle = g; o.fillRect(0, 0, 256, 256);
    } else if (which === 'circles') {
      o.fillStyle = '#0f172a'; o.fillRect(0, 0, 256, 256);
      for (let i = 0; i < 20; i++) {
        o.fillStyle = `hsl(${i * 18}, 70%, 60%)`;
        o.beginPath();
        o.arc(Math.random() * 256, Math.random() * 256, 10 + Math.random() * 40, 0, Math.PI * 2);
        o.fill();
      }
    } else {
      for (let i = 0; i < 256; i += 16) {
        o.fillStyle = i % 32 === 0 ? '#2563eb' : '#fbbf24';
        o.fillRect(i, 0, 16, 256);
      }
    }
  }

  function render() {
    const ps = +container.querySelector('#ip-p').value;
    const which = container.querySelector('#ip-img').value;
    container.querySelector('#ip-p-out').textContent = ps + 'px';
    drawImage(which);
    // draw patched version with gaps + patch numbers
    p.clearRect(0, 0, 256, 256);
    p.fillStyle = '#f3f4f6'; p.fillRect(0, 0, 256, 256);
    const gap = 2;
    const imgData = o.getImageData(0, 0, 256, 256);
    const origCvs = document.createElement('canvas');
    origCvs.width = 256; origCvs.height = 256;
    origCvs.getContext('2d').putImageData(imgData, 0, 0);
    const nPatches = Math.floor(256 / ps);
    for (let i = 0; i < nPatches; i++) {
      for (let j = 0; j < nPatches; j++) {
        p.drawImage(origCvs,
          j * ps, i * ps, ps, ps,
          j * ps + gap, i * ps + gap, ps - 2 * gap, ps - 2 * gap);
      }
    }
    const totalPatches = nPatches * nPatches;
    container.querySelector('#ip-out').textContent =
      `${nPatches}×${nPatches} = ${totalPatches} patches. Each patch (${ps}×${ps}×3) is flattened to ${ps*ps*3} numbers and linearly projected to an embedding. ` +
      `The transformer sees ${totalPatches} tokens — same as a sequence of ${totalPatches} words.`;
  }
  container.querySelector('#ip-p').addEventListener('input', render);
  container.querySelector('#ip-img').addEventListener('input', render);
  render();
});
