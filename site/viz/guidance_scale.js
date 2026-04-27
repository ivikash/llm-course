// viz/guidance_scale.js
// Simulate how classifier-free guidance shapes diffusion outputs.
// Shows stylized "images" — circles with prompt-matching features.

registerViz('guidance_scale', function (container) {
  container.innerHTML = `
    <p class="viz-title">Classifier-free guidance (CFG)</p>
    <p class="viz-sub">The CFG scale slider in every image generator. Low = loose interpretation. High = strict adherence (but can "fry").</p>
    <div class="viz-controls">
      <label>Prompt: <select id="gs-p">
        <option value="cat">a blue cat</option>
        <option value="sunset">orange sunset</option>
        <option value="forest">dense forest</option>
      </select></label>
      <label>Guidance scale: <input id="gs-s" type="range" min="0" max="20" step="0.5" value="7.5"><span id="gs-s-out">7.5</span></label>
    </div>
    <canvas id="gs-canvas" width="600" height="200" style="background:#fff;border:1px solid #d1d5db;border-radius:4px"></canvas>
    <p class="viz-readout" id="gs-out"></p>
  `;
  const cvs = container.querySelector('#gs-canvas');
  const ctx = cvs.getContext('2d');

  function render() {
    const prompt = container.querySelector('#gs-p').value;
    const scale = parseFloat(container.querySelector('#gs-s').value);
    container.querySelector('#gs-s-out').textContent = scale.toFixed(1);

    // Draw 3 sample images at scale 1, scale chosen, scale 15 for comparison
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, 600, 200);
    const labels = [`scale 1 (loose)`, `scale ${scale.toFixed(1)} (yours)`, 'scale 15 (strict)'];
    const scales = [1, scale, 15];
    for (let i = 0; i < 3; i++) {
      const x = 100 + i * 200;
      const y = 100;
      const s = scales[i];
      // base: noisy random blob
      ctx.save();
      ctx.beginPath();
      ctx.arc(x, y, 60, 0, Math.PI * 2);
      ctx.clip();
      // fill noise
      for (let j = 0; j < 200; j++) {
        ctx.fillStyle = `rgba(${Math.random()*255|0},${Math.random()*255|0},${Math.random()*255|0},0.3)`;
        ctx.fillRect(x - 60 + Math.random() * 120, y - 60 + Math.random() * 120, 4, 4);
      }
      // prompt-matching overlay with opacity proportional to scale
      const strength = Math.min(1, s / 10);
      if (prompt === 'cat') {
        ctx.fillStyle = `rgba(59,130,246,${strength})`;
        ctx.fillRect(x-40, y-30, 80, 60);
        ctx.fillStyle = `rgba(30,58,138,${strength})`;
        ctx.beginPath(); ctx.arc(x-25, y-5, 6, 0, Math.PI*2); ctx.fill();
        ctx.beginPath(); ctx.arc(x+25, y-5, 6, 0, Math.PI*2); ctx.fill();
      } else if (prompt === 'sunset') {
        const grd = ctx.createLinearGradient(x, y-60, x, y+60);
        grd.addColorStop(0, `rgba(251,146,60,${strength})`);
        grd.addColorStop(1, `rgba(127,29,29,${strength})`);
        ctx.fillStyle = grd;
        ctx.fillRect(x-60, y-60, 120, 120);
        ctx.fillStyle = `rgba(254,215,170,${strength})`;
        ctx.beginPath(); ctx.arc(x, y, 20, 0, Math.PI*2); ctx.fill();
      } else {
        for (let j = 0; j < 20; j++) {
          ctx.fillStyle = `rgba(${20+Math.random()*60|0},${100+Math.random()*50|0},${20+Math.random()*60|0},${strength})`;
          ctx.fillRect(x-60+j*6, y+20+Math.sin(j)*10, 5, -50 - Math.random()*30);
        }
      }
      // over-saturation for high scale
      if (s > 12) {
        ctx.fillStyle = `rgba(255,255,255,${(s - 12) / 16})`;
        ctx.fillRect(x-60, y-60, 120, 120);
      }
      ctx.restore();
      ctx.strokeStyle = '#6b7280';
      ctx.beginPath();
      ctx.arc(x, y, 60, 0, Math.PI*2);
      ctx.stroke();
      ctx.fillStyle = '#374151';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(labels[i], x, y + 82);
    }

    let desc = '';
    if (scale < 3) desc = 'Very loose — output is interpretive, sometimes weird.';
    else if (scale < 8) desc = 'Balanced — best for most prompts.';
    else if (scale < 14) desc = 'Strict adherence — looks polished but less creative.';
    else desc = 'Over-saturated ("fried") — may look artificial.';
    container.querySelector('#gs-out').textContent = desc;
  }
  container.querySelector('#gs-p').addEventListener('change', render);
  container.querySelector('#gs-s').addEventListener('input', render);
  render();
});
