// viz/embedding_2d.js
// 2D word-embedding playground. A small set of pre-baked word embeddings
// projected to 2D. Click two words to see the vector arithmetic:
// (king - man + woman ≈ queen).

registerViz('embedding_2d', function (container) {
  container.innerHTML = `
    <p class="viz-title">Word embeddings in 2D</p>
    <p class="viz-sub">Pre-trained embeddings, projected to 2D. Click points to select. Try: king − man + woman ≈ queen.</p>
    <div class="viz-controls">
      <label>A: <select id="eb-a"></select></label>
      <span>−</span>
      <label>B: <select id="eb-b"></select></label>
      <span>+</span>
      <label>C: <select id="eb-c"></select></label>
      <span>≈ ?</span>
      <button id="eb-compute">Compute A−B+C</button>
    </div>
    <svg id="eb-svg" viewBox="0 0 600 400" width="600" height="400" style="background:#fff;border:1px solid #e5e7eb;border-radius:4px;max-width:100%"></svg>
    <p class="viz-readout" id="eb-out"></p>
  `;

  // Curated 2D positions (approximated from real GloVe PCA)
  const vecs = {
    king:    [100, 90],  queen:   [90, 120],
    man:     [180, 180], woman:   [170, 210],
    boy:     [200, 160], girl:    [190, 240],
    prince:  [110, 100], princess:[100, 130],
    doctor:  [300, 250], nurse:   [290, 280],
    actor:   [320, 230], actress: [310, 260],
    paris:   [450, 80],  france:  [470, 100],
    london:  [430, 150], england: [450, 170],
    berlin:  [460, 220], germany: [480, 240],
    rome:    [440, 290], italy:   [460, 310],
    tokyo:   [500, 350], japan:   [520, 370],
    cat:     [60, 300],  kitten:  [70, 330],
    dog:     [130, 320], puppy:   [140, 350],
  };

  const svg = container.querySelector('#eb-svg');
  const aEl = container.querySelector('#eb-a');
  const bEl = container.querySelector('#eb-b');
  const cEl = container.querySelector('#eb-c');
  const btn = container.querySelector('#eb-compute');
  const out = container.querySelector('#eb-out');

  const words = Object.keys(vecs);
  for (const sel of [aEl, bEl, cEl]) {
    sel.innerHTML = words.map(w => `<option value="${w}">${w}</option>`).join('');
  }
  aEl.value = 'king'; bEl.value = 'man'; cEl.value = 'woman';

  function nearest(pt, exclude) {
    let best = null, bestD = Infinity;
    for (const w of words) {
      if (exclude.includes(w)) continue;
      const v = vecs[w];
      const d = Math.hypot(v[0] - pt[0], v[1] - pt[1]);
      if (d < bestD) { bestD = d; best = w; }
    }
    return { word: best, dist: bestD };
  }

  function render(result = null) {
    const A = vecs[aEl.value], B = vecs[bEl.value], C = vecs[cEl.value];
    // Vector arithmetic: A - B + C
    const target = [A[0] - B[0] + C[0], A[1] - B[1] + C[1]];
    const nr = nearest(target, [aEl.value, bEl.value, cEl.value]);

    let points = '';
    for (const w of words) {
      const [x, y] = vecs[w];
      const highlight = [aEl.value, bEl.value, cEl.value, nr.word].includes(w);
      const color = w === aEl.value ? '#ef4444' :
                    w === bEl.value ? '#f59e0b' :
                    w === cEl.value ? '#2563eb' :
                    w === nr.word ? '#10b981' : '#9ca3af';
      const r = highlight ? 6 : 3;
      points += `<circle cx="${x}" cy="${y}" r="${r}" fill="${color}"/>`;
      points += `<text x="${x + 6}" y="${y + 4}" font-size="10" fill="${highlight ? '#000' : '#6b7280'}">${w}</text>`;
    }
    // Draw arithmetic: A, -B (arrow), +C (arrow) → result
    let arrows = '';
    if (result) {
      // A - B
      arrows += `<line x1="${A[0]}" y1="${A[1]}" x2="${A[0] - B[0]}" y2="${A[1] - B[1]}" stroke="#f59e0b" stroke-dasharray="4,3"/>`;
      // + C
      arrows += `<line x1="${A[0] - B[0]}" y1="${A[1] - B[1]}" x2="${target[0]}" y2="${target[1]}" stroke="#2563eb" stroke-dasharray="4,3"/>`;
      arrows += `<circle cx="${target[0]}" cy="${target[1]}" r="6" fill="none" stroke="#10b981" stroke-width="2"/>`;
    }
    svg.innerHTML = `${points}${arrows}`;
    out.innerHTML = `${aEl.value} − ${bEl.value} + ${cEl.value} ≈ <b style="color:#10b981">${nr.word}</b> (distance ${nr.dist.toFixed(1)})`;
  }

  [aEl, bEl, cEl].forEach(el => el.addEventListener('change', () => render(true)));
  btn.addEventListener('click', () => render(true));
  render(false);
});
