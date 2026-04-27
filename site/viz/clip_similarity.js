// viz/clip_similarity.js
// Shows CLIP's contrastive training: a batch of image/caption pairs,
// similarity matrix that should be diagonal-heavy.

registerViz('clip_similarity', function (container) {
  container.innerHTML = `
    <p class="viz-title">CLIP: image ⟷ text similarity</p>
    <p class="viz-sub">Each row = one image, each column = one caption. Diagonal cells should be bright (matched pairs). Off-diagonal should be dark. Training pushes toward this.</p>
    <div class="viz-controls">
      <label>Training progress: <input id="cs-p" type="range" min="0" max="100" value="0"><span id="cs-p-out">0%</span></label>
      <button id="cs-train">▶ Train</button>
    </div>
    <div id="cs-body" style="display:flex;gap:16px;margin-top:12px;flex-wrap:wrap;justify-content:center"></div>
    <p class="viz-readout" id="cs-out"></p>
  `;
  const items = [
    { emoji: '🐕', caption: 'a dog' },
    { emoji: '🐈', caption: 'a cat' },
    { emoji: '🚗', caption: 'a car' },
    { emoji: '🏠', caption: 'a house' },
    { emoji: '🌳', caption: 'a tree' },
    { emoji: '🍕', caption: 'a pizza' },
    { emoji: '⚽', caption: 'a ball' },
  ];
  const N = items.length;

  function render() {
    const p = +container.querySelector('#cs-p').value / 100;
    container.querySelector('#cs-p-out').textContent = (p * 100).toFixed(0) + '%';
    // Similarity matrix: diagonal strengthens from random → 1, off-diag → 0
    const sim = Array.from({length: N}, (_, i) =>
      Array.from({length: N}, (_, j) => {
        const r = (Math.sin(i*23+j*7)*43758.5453 % 1 + 1) % 1;  // pseudo-random stable seed
        if (i === j) return 0.3 + 0.7 * p;              // diagonal grows
        return 0.3 + 0.4 * r * (1 - p);                   // off-diagonal shrinks
      })
    );
    const cellSize = 50;
    const W = (N + 1) * cellSize;
    const H = (N + 1) * cellSize;

    let html = `<div><svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" style="background:#fff;border:1px solid #e5e7eb;border-radius:4px">`;
    // column headers (captions)
    for (let j = 0; j < N; j++) {
      html += `<text x="${(j+1)*cellSize + cellSize/2}" y="${cellSize - 8}" font-size="10" text-anchor="middle" fill="#374151">${items[j].caption}</text>`;
    }
    // cells
    for (let i = 0; i < N; i++) {
      // row label (emoji)
      html += `<text x="${cellSize - 4}" y="${(i+1)*cellSize + cellSize/2 + 6}" font-size="24" text-anchor="end">${items[i].emoji}</text>`;
      for (let j = 0; j < N; j++) {
        const v = sim[i][j];
        const t = Math.max(0, Math.min(1, v));
        const g = Math.floor(20 + t * 180);
        const fill = `rgb(30, ${g}, 30)`;
        const border = i === j ? '#fbbf24' : '#fff';
        html += `<rect x="${(j+1)*cellSize + 2}" y="${(i+1)*cellSize + 2}" width="${cellSize-4}" height="${cellSize-4}" fill="${fill}" stroke="${border}" stroke-width="${i===j?2:1}"/>`;
        html += `<text x="${(j+1)*cellSize + cellSize/2}" y="${(i+1)*cellSize + cellSize/2 + 4}" font-size="10" text-anchor="middle" fill="#fff">${v.toFixed(2)}</text>`;
      }
    }
    html += `</svg></div>`;
    container.querySelector('#cs-body').innerHTML = html;
    const diagAvg = Array.from({length: N}, (_, i) => sim[i][i]).reduce((a,b)=>a+b,0) / N;
    const offAvg = sim.flatMap((row, i) => row.filter((_, j) => i !== j)).reduce((a,b)=>a+b,0) / (N*N - N);
    container.querySelector('#cs-out').innerHTML =
      `Diagonal (matched pairs): <b style="color:#10b981">${diagAvg.toFixed(3)}</b>. ` +
      `Off-diagonal (mismatched): <b style="color:#dc2626">${offAvg.toFixed(3)}</b>. ` +
      `Gap: ${(diagAvg - offAvg).toFixed(3)}. This gap IS what CLIP training maximizes (InfoNCE loss).`;
  }
  async function train() {
    const slider = container.querySelector('#cs-p');
    for (let p = 0; p <= 100; p += 2) {
      slider.value = p;
      render();
      await new Promise(r => setTimeout(r, 60));
    }
  }
  container.querySelector('#cs-p').addEventListener('input', render);
  container.querySelector('#cs-train').addEventListener('click', train);
  render();
});
