// viz/lora_adapter.js
// Show how a LoRA adapter A·B (low-rank) adds to a frozen W.

registerViz('lora_adapter', function (container) {
  container.innerHTML = `
    <p class="viz-title">LoRA: low-rank adapters</p>
    <p class="viz-sub">Freeze the huge weight W. Train only a thin A (d×r) and B (r×d). The update is W + A·B. When r≪d, the update has few trainable parameters.</p>
    <div class="viz-controls">
      <label>Matrix d: <input id="la-d" type="range" min="64" max="4096" step="64" value="1024"><span id="la-d-out">1024</span></label>
      <label>Rank r: <input id="la-r" type="range" min="1" max="128" value="8"><span id="la-r-out">8</span></label>
    </div>
    <svg id="la-svg" viewBox="0 0 760 260" width="760" height="260" style="background:#fff;border:1px solid #e5e7eb;border-radius:4px;max-width:100%;margin-top:12px"></svg>
    <p class="viz-readout" id="la-out"></p>
  `;

  function render() {
    const d = +container.querySelector('#la-d').value;
    const r = +container.querySelector('#la-r').value;
    container.querySelector('#la-d-out').textContent = d;
    container.querySelector('#la-r-out').textContent = r;

    const svg = container.querySelector('#la-svg');
    // Represent matrices as rectangles, size proportional to sqrt of params
    const scale = Math.min(160 / d, 160 / 128);
    const W_side = d * scale;
    const A_w = r * scale, A_h = d * scale;
    const B_w = d * scale, B_h = r * scale;
    const yMid = 120;

    let html = '';
    // W (frozen)
    const Wx = 60;
    html += `<rect x="${Wx}" y="${yMid - W_side/2}" width="${W_side}" height="${W_side}" fill="#e5e7eb" stroke="#6b7280"/>`;
    html += `<text x="${Wx + W_side/2}" y="${yMid - W_side/2 - 8}" font-size="11" text-anchor="middle" fill="#6b7280">W (frozen)</text>`;
    html += `<text x="${Wx + W_side/2}" y="${yMid + 4}" font-size="11" text-anchor="middle" fill="#374151" font-family="ui-monospace,monospace">${d}×${d}</text>`;
    html += `<text x="${Wx + W_side/2}" y="${yMid + 20}" font-size="10" text-anchor="middle" fill="#9ca3af">${(d*d).toLocaleString()} params</text>`;

    // + sign
    html += `<text x="${Wx + W_side + 20}" y="${yMid + 4}" font-size="22" text-anchor="middle" fill="#374151">+</text>`;

    // A
    const Ax = Wx + W_side + 60;
    html += `<rect x="${Ax}" y="${yMid - A_h/2}" width="${Math.max(8, A_w)}" height="${A_h}" fill="#fbbf24" stroke="#d97706"/>`;
    html += `<text x="${Ax + Math.max(4, A_w/2)}" y="${yMid - A_h/2 - 8}" font-size="11" text-anchor="middle" fill="#92400e">A</text>`;
    html += `<text x="${Ax + Math.max(4, A_w/2)}" y="${yMid + A_h/2 + 14}" font-size="9" text-anchor="middle" fill="#92400e" font-family="ui-monospace,monospace">${d}×${r}</text>`;

    // · sign
    html += `<text x="${Ax + Math.max(8, A_w) + 15}" y="${yMid + 4}" font-size="18" text-anchor="middle" fill="#374151">·</text>`;

    // B
    const Bx = Ax + Math.max(8, A_w) + 35;
    html += `<rect x="${Bx}" y="${yMid - Math.max(8, B_h/2)}" width="${B_w}" height="${Math.max(8, B_h)}" fill="#60a5fa" stroke="#2563eb"/>`;
    html += `<text x="${Bx + B_w/2}" y="${yMid - Math.max(8, B_h/2) - 4}" font-size="11" text-anchor="middle" fill="#1e3a8a">B</text>`;
    html += `<text x="${Bx + B_w/2}" y="${yMid + Math.max(8, B_h/2) + 14}" font-size="9" text-anchor="middle" fill="#1e3a8a" font-family="ui-monospace,monospace">${r}×${d}</text>`;

    svg.innerHTML = html;

    const fullParams = d * d;
    const loraParams = 2 * d * r;
    const reduction = fullParams / loraParams;
    const fileSize = (loraParams * 2 / 1e6).toFixed(1);
    container.querySelector('#la-out').innerHTML =
      `Full fine-tune: <b>${fullParams.toLocaleString()}</b> params to train. ` +
      `LoRA (rank ${r}): <b>${loraParams.toLocaleString()}</b> params — a <b>${reduction.toFixed(0)}× reduction</b>. ` +
      `Adapter file size (fp16): <b>${fileSize} MB</b>. ` +
      `At inference, either merge (W + A·B stored) or keep separate for per-task swap. Typical r for LLMs: 8-64.`;
  }
  container.querySelector('#la-d').addEventListener('input', render);
  container.querySelector('#la-r').addEventListener('input', render);
  render();
});
