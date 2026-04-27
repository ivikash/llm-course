// viz/memmap_batching.js
// Animate random-offset batch sampling from a memmapped token stream.

registerViz('memmap_batching', function (container) {
  container.innerHTML = `
    <p class="viz-title">How nanoGPT reads batches from disk</p>
    <p class="viz-sub">train.bin is a giant array of uint16 token IDs. Each step: pick random offsets, slice block_size tokens, stack into a batch. No DataLoader, no epochs.</p>
    <div class="viz-controls">
      <label>Batch size: <input id="mb-b" type="range" min="2" max="8" value="4"><span id="mb-b-out">4</span></label>
      <label>Block size: <input id="mb-bl" type="range" min="8" max="32" value="16"><span id="mb-bl-out">16</span></label>
      <button id="mb-new">New batch</button>
    </div>
    <div>
      <div style="font-size:11px;color:#6b7280;margin-bottom:4px">train.bin (9B tokens)</div>
      <div id="mb-file" style="font-family:ui-monospace,monospace;font-size:11px;white-space:nowrap;overflow:hidden;background:#fafafa;border:1px solid #e5e7eb;border-radius:4px;padding:6px"></div>
    </div>
    <div style="margin-top:12px">
      <div style="font-size:11px;color:#6b7280;margin-bottom:4px">x (inputs) — batch of block_size slices</div>
      <div id="mb-x" style="font-family:ui-monospace,monospace;font-size:11px;display:flex;flex-direction:column;gap:4px"></div>
    </div>
    <div style="margin-top:12px">
      <div style="font-size:11px;color:#6b7280;margin-bottom:4px">y (targets = x shifted by 1)</div>
      <div id="mb-y" style="font-family:ui-monospace,monospace;font-size:11px;display:flex;flex-direction:column;gap:4px"></div>
    </div>
    <p class="viz-readout" id="mb-out"></p>
  `;
  const FILE_LEN = 120;
  // Make a deterministic-ish fake token stream
  const tokens = Array.from({ length: FILE_LEN }, (_, i) => (i * 37 + 11) % 100 + 200);
  const colors = ['#fecaca', '#fed7aa', '#fef3c7', '#bbf7d0', '#bfdbfe', '#e9d5ff', '#fce7f3', '#d1fae5'];

  function newBatch() {
    const B = +container.querySelector('#mb-b').value;
    const BL = +container.querySelector('#mb-bl').value;
    container.querySelector('#mb-b-out').textContent = B;
    container.querySelector('#mb-bl-out').textContent = BL;

    const offsets = [];
    while (offsets.length < B) {
      const o = Math.floor(Math.random() * (FILE_LEN - BL - 1));
      if (!offsets.includes(o)) offsets.push(o);
    }
    offsets.sort((a,b) => a - b);

    // Render file with highlighted slices
    let fileHtml = '';
    for (let i = 0; i < FILE_LEN; i++) {
      let bg = '#fff', color = '#6b7280';
      for (let k = 0; k < offsets.length; k++) {
        const o = offsets[k];
        if (i >= o && i < o + BL) { bg = colors[k % colors.length]; color = '#111'; break; }
      }
      fileHtml += `<span style="background:${bg};color:${color};padding:2px 4px;margin:0 1px;border-radius:2px">${tokens[i]}</span>`;
    }
    container.querySelector('#mb-file').innerHTML = fileHtml;

    // Render x and y
    let xHtml = '', yHtml = '';
    offsets.forEach((o, k) => {
      const xSlice = tokens.slice(o, o + BL);
      const ySlice = tokens.slice(o + 1, o + 1 + BL);
      const bg = colors[k % colors.length];
      xHtml += `<div><span style="color:#9ca3af">[${o}]</span> ` + xSlice.map(t => `<span style="background:${bg};padding:2px 4px;margin:0 1px;border-radius:2px">${t}</span>`).join('') + '</div>';
      yHtml += `<div><span style="color:#9ca3af">[${o+1}]</span> ` + ySlice.map(t => `<span style="background:${bg};padding:2px 4px;margin:0 1px;border-radius:2px">${t}</span>`).join('') + '</div>';
    });
    container.querySelector('#mb-x').innerHTML = xHtml;
    container.querySelector('#mb-y').innerHTML = yHtml;
    container.querySelector('#mb-out').textContent =
      `Batch shape: x=(${B}, ${BL}) int16. y is x shifted by 1. No 'epoch' — just keep sampling until max_iters. This is the whole data loader.`;
  }
  container.querySelector('#mb-new').addEventListener('click', newBatch);
  container.querySelector('#mb-b').addEventListener('input', newBatch);
  container.querySelector('#mb-bl').addEventListener('input', newBatch);
  newBatch();
});
