// viz/attention_math.js
// Full Q, K, V, softmax, @V pipeline on a toy example. User sees each
// tensor and can change the numbers.

registerViz('attention_math', function (container) {
  container.innerHTML = `
    <p class="viz-title">Attention, mathematically (step by step)</p>
    <p class="viz-sub">A tiny 4-token, 3-dim example. See Q, K, V, scores, softmax, and output tensors. Each cell colored by value. Randomize to see different patterns.</p>
    <div class="viz-controls">
      <button id="am-rand">Randomize</button>
      <label>Seq len: <input id="am-T" type="range" min="2" max="8" value="4"><span id="am-T-out">4</span></label>
      <label>Dim: <input id="am-d" type="range" min="2" max="8" value="3"><span id="am-d-out">3</span></label>
      <label><input type="checkbox" id="am-mask" checked> Causal mask</label>
    </div>
    <div id="am-body" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-top:12px"></div>
    <p class="viz-readout" id="am-out"></p>
  `;

  let T = 4, d = 3;
  let Q, K, V;

  function randn() { return (Math.random() - 0.5) * 2; }
  function randomize() {
    T = parseInt(container.querySelector('#am-T').value);
    d = parseInt(container.querySelector('#am-d').value);
    container.querySelector('#am-T-out').textContent = T;
    container.querySelector('#am-d-out').textContent = d;
    Q = Array.from({ length: T }, () => Array.from({ length: d }, randn));
    K = Array.from({ length: T }, () => Array.from({ length: d }, randn));
    V = Array.from({ length: T }, () => Array.from({ length: d }, randn));
    render();
  }

  function matmul(A, B_transposed_row_of_B) {  // A:(T×d) @ B^T where B:(T×d) so result: T×T
    const M = Array.from({ length: A.length }, () => new Array(B_transposed_row_of_B.length).fill(0));
    for (let i = 0; i < A.length; i++)
      for (let j = 0; j < B_transposed_row_of_B.length; j++)
        for (let k = 0; k < A[0].length; k++)
          M[i][j] += A[i][k] * B_transposed_row_of_B[j][k];
    return M;
  }
  function matmulAB(A, B) {  // A:(m×k) @ B:(k×n)
    const m = A.length, k = A[0].length, n = B[0].length;
    const M = Array.from({ length: m }, () => new Array(n).fill(0));
    for (let i = 0; i < m; i++)
      for (let j = 0; j < n; j++)
        for (let p = 0; p < k; p++)
          M[i][j] += A[i][p] * B[p][j];
    return M;
  }
  function softmaxRows(S, mask) {
    return S.map((row, i) => {
      let logits = row.slice();
      if (mask) for (let j = i + 1; j < logits.length; j++) logits[j] = -Infinity;
      const maxL = Math.max(...logits);
      const exps = logits.map(l => Math.exp(l - maxL));
      const sum = exps.reduce((a, b) => a + b, 0);
      return exps.map(e => e / sum);
    });
  }

  function cell(v, vmin, vmax, fmt) {
    const t = (v - vmin) / (vmax - vmin || 1);
    const r = Math.floor(255 - t * 220);
    const g = Math.floor(255 - Math.abs(0.5 - t) * 255);
    const b = Math.floor(30 + t * 180);
    return `<td style="padding:4px 6px;background:rgb(${r},${g},${b});color:${t > 0.5 ? '#fff' : '#111'};text-align:center;font-family:ui-monospace,monospace;font-size:11px">${(fmt || (x => x.toFixed(2)))(v)}</td>`;
  }
  function matHtml(M, title, fmt) {
    const flat = M.flat();
    const vmin = Math.min(...flat.filter(x => isFinite(x)));
    const vmax = Math.max(...flat.filter(x => isFinite(x)));
    return `<div style="text-align:center"><div style="font-size:11px;color:#6b7280;margin-bottom:4px">${title}</div>` +
      `<table style="border-collapse:collapse;margin:0 auto"><tbody>` +
      M.map(row => `<tr>${row.map(v => cell(v, vmin, vmax, fmt)).join('')}</tr>`).join('') +
      `</tbody></table></div>`;
  }

  function render() {
    const mask = container.querySelector('#am-mask').checked;
    // compute attention pipeline
    const scores = matmul(Q, K).map(row => row.map(v => v / Math.sqrt(d)));
    const weights = softmaxRows(scores, mask);
    const output = matmulAB(weights, V);

    const body = container.querySelector('#am-body');
    body.innerHTML = `
      ${matHtml(Q, `Q (T=${T}, d=${d})`)}
      ${matHtml(K, `K (T=${T}, d=${d})`)}
      ${matHtml(V, `V (T=${T}, d=${d})`)}
      ${matHtml(scores, `Q·K⊤/√d (T×T)`)}
      ${matHtml(weights, `softmax → weights`, v => (v * 100).toFixed(0) + '%')}
      ${matHtml(output, `weights @ V → output (T×d)`)}
    `;
    container.querySelector('#am-out').textContent =
      `Each output row = weighted sum of V rows, weighted by how much each query matches each key. ` +
      (mask ? 'Causal mask applied (upper-triangle = -inf).' : 'No mask — bidirectional.');
  }

  container.querySelector('#am-rand').addEventListener('click', randomize);
  container.querySelector('#am-T').addEventListener('input', randomize);
  container.querySelector('#am-d').addEventListener('input', randomize);
  container.querySelector('#am-mask').addEventListener('change', render);
  randomize();
});
