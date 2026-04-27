// viz/matmul_animated.js
// Animated matrix multiplication: user picks row/col, watches
// the dot product light up element by element.

registerViz('matmul_animated', function (container) {
  container.innerHTML = `
    <p class="viz-title">Matrix multiplication, step by step</p>
    <p class="viz-sub">Click a cell in C. Watch row i of A dotted with column j of B to produce C[i,j].</p>
    <div id="mm-root" style="display:flex;gap:24px;justify-content:center;align-items:center;flex-wrap:wrap;font-family:ui-monospace,monospace;font-size:14px"></div>
    <div class="viz-controls">
      <button id="mm-rand">Randomize</button>
      <span id="mm-result" style="margin-left:auto"></span>
    </div>
  `;

  const m = 3, k = 4, n = 3; // A:(3×4) B:(4×3) → C:(3×3)
  let A = [], B = [];
  let selI = 0, selJ = 0, step = 0, animating = false;

  function randomize() {
    A = Array.from({ length: m }, () =>
      Array.from({ length: k }, () => (Math.random() * 10 - 5) | 0));
    B = Array.from({ length: k }, () =>
      Array.from({ length: n }, () => (Math.random() * 10 - 5) | 0));
  }

  function computeC() {
    const C = Array.from({ length: m }, () => new Array(n).fill(0));
    for (let i = 0; i < m; i++)
      for (let j = 0; j < n; j++)
        for (let p = 0; p < k; p++)
          C[i][j] += A[i][p] * B[p][j];
    return C;
  }

  function render() {
    const C = computeC();
    const root = container.querySelector('#mm-root');
    root.innerHTML = `
      <div>
        <div style="font-size:12px;color:#6b7280;margin-bottom:4px">A (${m}×${k})</div>
        <table>${A.map((row, i) => `<tr>${row.map((v, p) =>
          `<td class="mm-a" data-i="${i}" data-p="${p}" style="padding:8px 12px;border:1px solid #d1d5db;${
            i === selI ? 'background:#fde68a' : 'background:#fff'
          }${p === step && i === selI ? 'outline:3px solid #ea580c;outline-offset:-3px;' : ''}">${v}</td>`).join('')}</tr>`).join('')}</table>
      </div>
      <div style="font-size:28px">×</div>
      <div>
        <div style="font-size:12px;color:#6b7280;margin-bottom:4px">B (${k}×${n})</div>
        <table>${B.map((row, p) => `<tr>${row.map((v, j) =>
          `<td class="mm-b" data-p="${p}" data-j="${j}" style="padding:8px 12px;border:1px solid #d1d5db;${
            j === selJ ? 'background:#bfdbfe' : 'background:#fff'
          }${p === step && j === selJ ? 'outline:3px solid #ea580c;outline-offset:-3px;' : ''}">${v}</td>`).join('')}</tr>`).join('')}</table>
      </div>
      <div style="font-size:28px">=</div>
      <div>
        <div style="font-size:12px;color:#6b7280;margin-bottom:4px">C (${m}×${n})</div>
        <table>${C.map((row, i) => `<tr>${row.map((v, j) =>
          `<td class="mm-c" data-i="${i}" data-j="${j}" style="padding:8px 12px;border:1px solid #d1d5db;cursor:pointer;${
            i === selI && j === selJ ? 'background:#86efac;font-weight:bold' : 'background:#fff'
          }">${v}</td>`).join('')}</tr>`).join('')}</table>
      </div>
    `;
    // read-out live during animation
    const partialSum = A[selI].slice(0, step).reduce((s, v, p) => s + v * B[p][selJ], 0);
    const finalVal = C[selI][selJ];
    const res = container.querySelector('#mm-result');
    if (animating && step < k) {
      res.innerHTML = `C[${selI},${selJ}] partial = ` +
        A[selI].slice(0, step + 1).map((a, p) => `${a}×${B[p][selJ]}`).join(' + ') +
        ` = <b>${partialSum + A[selI][step] * B[step][selJ]}</b>`;
    } else {
      res.innerHTML = `C[${selI},${selJ}] = ` +
        A[selI].map((a, p) => `${a}×${B[p][selJ]}`).join(' + ') + ` = <b>${finalVal}</b>`;
    }
    container.querySelectorAll('.mm-c').forEach(td => {
      td.addEventListener('click', () => {
        selI = +td.dataset.i; selJ = +td.dataset.j;
        animateStep();
      });
    });
  }

  async function animateStep() {
    animating = true;
    for (step = 0; step < k; step++) {
      render();
      await new Promise(r => setTimeout(r, 400));
    }
    step = k;
    animating = false;
    render();
  }

  container.querySelector('#mm-rand').addEventListener('click', () => {
    randomize(); step = k; render();
  });

  randomize();
  animateStep();
});
