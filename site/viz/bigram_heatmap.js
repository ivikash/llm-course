// viz/bigram_heatmap.js
// Renders a bigram transition probability heatmap as an SVG/canvas.
// Shows which characters follow which (bright cells = frequent pair).

registerViz('bigram_heatmap', function (container) {
  container.innerHTML = `
    <p class="viz-title">Bigram transition heatmap</p>
    <p class="viz-sub">Y axis = current character, X axis = next character. Brighter = higher probability. Hover a cell for details.</p>
    <canvas id="bh-canvas" width="640" height="640" style="image-rendering:pixelated;cursor:crosshair"></canvas>
    <p class="viz-readout" id="bh-readout">Hover to inspect.</p>
  `;

  const sample = `ROMEO: But soft, what light through yonder window breaks? It is the east, and Juliet is the sun.
JULIET: O Romeo, Romeo! wherefore art thou Romeo? Deny thy father and refuse thy name.
MERCUTIO: A plague o' both your houses! They have made worms' meat of me.
HAMLET: To be, or not to be, that is the question: whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune.
MACBETH: Is this a dagger which I see before me, the handle toward my hand?
KING LEAR: Blow, winds, and crack your cheeks! Rage! Blow! You cataracts and hurricanoes.
PORTIA: The quality of mercy is not strained. It droppeth as the gentle rain from heaven.
BENEDICK: When I said I would die a bachelor, I did not think I should live till I were married.
OTHELLO: Speak of me as I am. Nothing extenuate, nor set down aught in malice.`;

  // Build matrix
  const charSet = new Set(sample);
  const chars = [...charSet].sort();
  const idx = Object.fromEntries(chars.map((c, i) => [c, i]));
  const N = chars.length;
  const counts = Array.from({ length: N }, () => new Array(N).fill(0));
  for (let i = 0; i < sample.length - 1; i++) {
    const a = idx[sample[i]], b = idx[sample[i + 1]];
    counts[a][b]++;
  }
  // Row-normalize to probabilities
  const probs = counts.map(row => {
    const sum = row.reduce((a, b) => a + b, 0) || 1;
    return row.map(v => v / sum);
  });

  const cvs = container.querySelector('#bh-canvas');
  const ctx = cvs.getContext('2d');
  const readout = container.querySelector('#bh-readout');
  const cellSize = Math.floor(cvs.width / N);
  const W = cellSize * N;
  cvs.width = W; cvs.height = W;

  function draw() {
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, W, W);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const v = probs[i][j];
        // viridis-like: low=dark purple, high=yellow
        const t = Math.pow(v, 0.4);  // gamma for visibility
        const r = Math.floor(68 + 200 * t);
        const g = Math.floor(1 + 230 * t);
        const b = Math.floor(84 - 50 * t);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
      }
    }
    // labels (sparse)
    ctx.fillStyle = '#fff';
    ctx.font = `${Math.max(8, cellSize - 4)}px monospace`;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'center';
    for (let i = 0; i < N; i++) {
      const label = chars[i] === ' ' ? '·' : chars[i] === '\n' ? '↵' : chars[i];
      ctx.fillText(label, i * cellSize + cellSize / 2, 4);
      ctx.fillText(label, 4, i * cellSize + cellSize / 2);
    }
  }
  draw();

  cvs.addEventListener('mousemove', e => {
    const rect = cvs.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / rect.width * N);
    const y = Math.floor((e.clientY - rect.top) / rect.height * N);
    if (x < 0 || x >= N || y < 0 || y >= N) return;
    const from = chars[y] === '\n' ? '\\n' : chars[y];
    const to = chars[x] === '\n' ? '\\n' : chars[x];
    const p = probs[y][x];
    readout.textContent = `P( '${to}' | '${from}' ) = ${(p * 100).toFixed(1)}%   (seen ${counts[y][x]} times after '${from}')`;
  });
});
