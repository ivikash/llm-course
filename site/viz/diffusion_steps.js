// viz/diffusion_steps.js
// Diffusion forward + reverse process visualized on a canvas.
// Shows: start with image → add noise at each step → pure noise
// Then: denoise in reverse.

registerViz('diffusion_steps', function (container) {
  container.innerHTML = `
    <p class="viz-title">Diffusion: noise ⇄ image</p>
    <p class="viz-sub">Forward: gradually add noise to an image. Reverse: the model learns to remove noise. Play forward to see destruction; then reverse to see generation.</p>
    <div class="viz-controls">
      <label>Step: <input id="df-s" type="range" min="0" max="1000" step="10" value="0"><span id="df-s-out">0</span></label>
      <button id="df-forward">▶ Forward (add noise)</button>
      <button id="df-reverse">◀ Reverse (denoise)</button>
    </div>
    <canvas id="df-canvas" width="256" height="256" style="border:1px solid #d1d5db;border-radius:4px"></canvas>
    <p class="viz-readout" id="df-out"></p>
  `;
  const cvs = container.querySelector('#df-canvas');
  const ctx = cvs.getContext('2d');

  // Create a base "image"
  function drawBase() {
    const g = ctx.createLinearGradient(0, 0, 256, 256);
    g.addColorStop(0, '#fda4af'); g.addColorStop(0.5, '#fbbf24'); g.addColorStop(1, '#60a5fa');
    ctx.fillStyle = g; ctx.fillRect(0, 0, 256, 256);
    ctx.fillStyle = '#111';
    ctx.font = 'bold 48px serif';
    ctx.fillText('🐱', 100, 160);
    return ctx.getImageData(0, 0, 256, 256);
  }
  let baseImg = drawBase();

  function render(step) {
    const T = 1000;
    const alpha = 1 - step / T;   // signal amount
    const sqrtA = Math.sqrt(alpha);
    const sqrt1mA = Math.sqrt(1 - alpha);

    const out = ctx.createImageData(256, 256);
    for (let i = 0; i < baseImg.data.length; i += 4) {
      // Gaussian noise
      const n1 = (Math.random() + Math.random() + Math.random() - 1.5) * 100;
      const n2 = (Math.random() + Math.random() + Math.random() - 1.5) * 100;
      const n3 = (Math.random() + Math.random() + Math.random() - 1.5) * 100;
      out.data[i]     = Math.max(0, Math.min(255, baseImg.data[i]     * sqrtA + n1 * sqrt1mA + 128 * sqrt1mA));
      out.data[i + 1] = Math.max(0, Math.min(255, baseImg.data[i + 1] * sqrtA + n2 * sqrt1mA + 128 * sqrt1mA));
      out.data[i + 2] = Math.max(0, Math.min(255, baseImg.data[i + 2] * sqrtA + n3 * sqrt1mA + 128 * sqrt1mA));
      out.data[i + 3] = 255;
    }
    ctx.putImageData(out, 0, 0);
    container.querySelector('#df-s-out').textContent = step;
    container.querySelector('#df-out').textContent =
      `step=${step}, α=${alpha.toFixed(3)}, signal=${(sqrtA * 100).toFixed(1)}%, noise=${(sqrt1mA * 100).toFixed(1)}%`;
  }

  function anim(from, to) {
    const steps = Math.abs(to - from) < 10 ? Math.abs(to - from) : 40;
    const dir = to > from ? 1 : -1;
    let cur = from;
    const interval = setInterval(() => {
      cur += dir * (Math.abs(to - from) / steps);
      if ((dir > 0 && cur >= to) || (dir < 0 && cur <= to)) {
        cur = to;
        clearInterval(interval);
      }
      container.querySelector('#df-s').value = Math.round(cur);
      render(Math.round(cur));
    }, 50);
  }

  container.querySelector('#df-s').addEventListener('input', e => render(+e.target.value));
  container.querySelector('#df-forward').addEventListener('click', () => anim(+container.querySelector('#df-s').value, 1000));
  container.querySelector('#df-reverse').addEventListener('click', () => anim(+container.querySelector('#df-s').value, 0));
  render(0);
});
