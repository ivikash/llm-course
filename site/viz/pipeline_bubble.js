// viz/pipeline_bubble.js
// Classic pipeline parallelism: show the bubble, then micro-batching.

registerViz('pipeline_bubble', function (container) {
  container.innerHTML = `
    <p class="viz-title">Pipeline parallelism: the bubble problem</p>
    <p class="viz-sub">4 GPUs, 4 layers of a model. Naive: lots of idle time. Micro-batching: hides most of the bubble.</p>
    <div class="viz-controls">
      <label>Mode:
        <select id="pb-m">
          <option value="naive" selected>Naive pipeline</option>
          <option value="micro">Micro-batching (GPipe)</option>
          <option value="1f1b">1F1B interleaved</option>
        </select>
      </label>
      <label>Micro-batches: <input id="pb-k" type="range" min="1" max="8" value="4"><span id="pb-k-out">4</span></label>
    </div>
    <svg id="pb-svg" viewBox="0 0 800 200" width="800" height="200" style="background:#fff;border:1px solid #e5e7eb;border-radius:4px;max-width:100%;margin-top:12px"></svg>
    <p class="viz-readout" id="pb-out"></p>
  `;

  function render() {
    const mode = container.querySelector('#pb-m').value;
    const K = +container.querySelector('#pb-k').value;
    container.querySelector('#pb-k-out').textContent = K;

    const svg = container.querySelector('#pb-svg');
    const GPUs = 4;
    const cellW = mode === 'naive' ? 40 : 30;
    const rowH = 30;
    const yStart = 20;
    const xStart = 60;

    // Build schedule: each cell is {gpu, time, kind ('F' or 'B'), mb}
    const cells = [];
    if (mode === 'naive') {
      // One forward sweeps across 4 GPUs, then backward sweeps back
      for (let g = 0; g < GPUs; g++) cells.push({ gpu: g, time: g, kind: 'F', mb: 0 });
      for (let g = GPUs - 1; g >= 0; g--) cells.push({ gpu: g, time: GPUs + (GPUs - 1 - g), kind: 'B', mb: 0 });
    } else if (mode === 'micro') {
      // GPipe: all forwards of all micro-batches, then all backwards
      for (let m = 0; m < K; m++) {
        for (let g = 0; g < GPUs; g++) cells.push({ gpu: g, time: g + m, kind: 'F', mb: m });
      }
      const fEnd = K + GPUs - 1;
      for (let m = 0; m < K; m++) {
        for (let g = GPUs - 1; g >= 0; g--) cells.push({ gpu: g, time: fEnd + (GPUs - 1 - g) + m, kind: 'B', mb: m });
      }
    } else {
      // 1F1B: stagger fwd/bwd
      // Warmup: GPU g does (GPUs - g) forward passes first
      // Then alternating F/B
      let perGpu = Array.from({ length: GPUs }, () => []);
      // Fwd schedule
      for (let m = 0; m < K; m++) {
        for (let g = 0; g < GPUs; g++) {
          perGpu[g].push({ kind: 'F', mb: m });
        }
      }
      // Bwd schedule (reversed order of GPUs, same mb order)
      for (let m = 0; m < K; m++) {
        for (let g = GPUs - 1; g >= 0; g--) {
          perGpu[g].push({ kind: 'B', mb: m });
        }
      }
      // Now assign times — this is the simplified visual, each step 1 slot
      for (let g = 0; g < GPUs; g++) {
        for (let t = 0; t < perGpu[g].length; t++) {
          const offset = perGpu[g][t].kind === 'F' ? g : (GPUs - 1 - g);
          cells.push({ gpu: g, time: t + offset / 4, kind: perGpu[g][t].kind, mb: perGpu[g][t].mb });
        }
      }
    }

    const maxTime = Math.max(...cells.map(c => c.time)) + 1;
    let html = '';
    // axes
    for (let g = 0; g < GPUs; g++) {
      html += `<text x="40" y="${yStart + g * rowH + 20}" font-size="11" text-anchor="end" fill="#374151">GPU ${g}</text>`;
      html += `<line x1="${xStart}" y1="${yStart + g * rowH + 15}" x2="${xStart + maxTime * cellW + 10}" y2="${yStart + g * rowH + 15}" stroke="#e5e7eb"/>`;
    }
    // bubbles (idle cells)
    let bubbleCount = 0;
    const occupied = new Set();
    cells.forEach(c => occupied.add(`${c.gpu}:${Math.floor(c.time)}`));
    for (let g = 0; g < GPUs; g++) {
      for (let t = 0; t < maxTime; t++) {
        if (!occupied.has(`${g}:${t}`)) {
          html += `<rect x="${xStart + t * cellW}" y="${yStart + g * rowH + 2}" width="${cellW - 2}" height="${rowH - 6}" fill="#f3f4f6" stroke="#d1d5db" stroke-dasharray="2,2"/>`;
          bubbleCount++;
        }
      }
    }
    // work cells
    cells.forEach(c => {
      const colors = ['#3b82f6', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316', '#eab308'];
      const color = c.kind === 'F' ? colors[c.mb % colors.length] : '#7c3aed';
      const x = xStart + c.time * cellW;
      html += `<rect x="${x}" y="${yStart + c.gpu * rowH + 2}" width="${cellW - 2}" height="${rowH - 6}" fill="${color}" opacity="${c.kind === 'B' ? 0.7 : 1}" rx="2"/>`;
      html += `<text x="${x + cellW/2 - 1}" y="${yStart + c.gpu * rowH + 17}" font-size="9" text-anchor="middle" fill="#fff" font-weight="bold">${c.kind}${c.mb}</text>`;
    });
    svg.innerHTML = html;

    const total = GPUs * maxTime;
    const util = ((total - bubbleCount) / total * 100).toFixed(0);
    container.querySelector('#pb-out').innerHTML =
      `Schedule length: ${maxTime} time units. Idle cells: ${bubbleCount}. <b>GPU utilization: ${util}%</b>. ` +
      (mode === 'naive' ? 'Only 25% utilization — naive pipeline is a disaster.' :
       mode === 'micro' ? `Micro-batching pushes utilization up. With K=${K} micro-batches, bubble shrinks.` :
       '1F1B interleaves fwd and bwd to reduce peak memory while keeping utilization high. Used by Megatron-LM, DeepSpeed.');
  }

  container.querySelector('#pb-m').addEventListener('change', render);
  container.querySelector('#pb-k').addEventListener('input', render);
  render();
});
