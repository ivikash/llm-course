// viz/backprop_graph.js
// Compute graph of a 2-layer MLP. Forward values flow, then backward grads flow.

registerViz('backprop_graph', function (container) {
  container.innerHTML = `
    <p class="viz-title">Backprop: watch gradients flow</p>
    <p class="viz-sub">A 2-layer MLP on a tiny example. Forward pass fills values. Backward pass fills gradients (red numbers).</p>
    <div class="viz-controls">
      <label>x: <input id="bg-x" type="range" min="-3" max="3" step="0.1" value="1.5"><span id="bg-x-out">1.5</span></label>
      <label>target: <input id="bg-y" type="range" min="-3" max="3" step="0.1" value="2"><span id="bg-y-out">2.0</span></label>
      <button id="bg-fwd">▶ Forward</button>
      <button id="bg-bwd">◀ Backward</button>
      <button id="bg-reset">Reset</button>
    </div>
    <svg id="bg-svg" viewBox="0 0 800 340" width="800" height="340" style="background:#fff;border:1px solid #e5e7eb;border-radius:4px;max-width:100%"></svg>
    <p class="viz-readout" id="bg-out"></p>
  `;
  // Fixed weights for demo: h = relu(W1*x + b1); y = W2*h + b2; loss = (y - target)^2
  const W1 = 1.5, b1 = -0.5, W2 = 2.0, b2 = 0.3;

  const nodes = [
    { id: 'x',    x: 60,  y: 170, label: 'x' },
    { id: 'W1',   x: 60,  y: 80,  label: 'W1' },
    { id: 'b1',   x: 60,  y: 260, label: 'b1' },
    { id: 'z1',   x: 220, y: 170, label: 'z1 = W1·x + b1' },
    { id: 'h',    x: 370, y: 170, label: 'h = ReLU(z1)' },
    { id: 'W2',   x: 370, y: 80,  label: 'W2' },
    { id: 'b2',   x: 370, y: 260, label: 'b2' },
    { id: 'yhat', x: 540, y: 170, label: 'y_hat = W2·h + b2' },
    { id: 'tgt',  x: 540, y: 260, label: 'target' },
    { id: 'loss', x: 720, y: 170, label: '(y_hat − target)²' },
  ];
  const edges = [
    ['x','z1'], ['W1','z1'], ['b1','z1'],
    ['z1','h'],
    ['h','yhat'], ['W2','yhat'], ['b2','yhat'],
    ['yhat','loss'], ['tgt','loss'],
  ];

  let values = {};
  let grads = {};
  let phase = 'none';

  function forward() {
    const x = +container.querySelector('#bg-x').value;
    const target = +container.querySelector('#bg-y').value;
    values = {
      x, W1, b1, W2, b2, tgt: target,
      z1: W1 * x + b1,
    };
    values.h = Math.max(0, values.z1);
    values.yhat = W2 * values.h + b2;
    values.loss = (values.yhat - target) ** 2;
    grads = {};
    phase = 'forward';
    render();
  }
  function backward() {
    if (!values.loss) return;
    grads = {};
    // dloss/dyhat = 2(yhat - target)
    grads.yhat = 2 * (values.yhat - values.tgt);
    grads.W2 = grads.yhat * values.h;
    grads.b2 = grads.yhat;
    grads.h = grads.yhat * W2;
    grads.z1 = grads.h * (values.z1 > 0 ? 1 : 0);
    grads.W1 = grads.z1 * values.x;
    grads.b1 = grads.z1;
    grads.x = grads.z1 * W1;
    phase = 'backward';
    render();
  }

  function render() {
    let html = '';
    for (const [a, b] of edges) {
      const na = nodes.find(n => n.id === a), nb = nodes.find(n => n.id === b);
      const active = phase === 'forward' ? 'stroke="#60a5fa" stroke-width="2"' :
                     phase === 'backward' ? 'stroke="#dc2626" stroke-width="2"' :
                     'stroke="#d1d5db"';
      html += `<line x1="${na.x+60}" y1="${na.y}" x2="${nb.x-40}" y2="${nb.y}" ${active}/>`;
    }
    nodes.forEach(n => {
      const val = values[n.id] !== undefined ? values[n.id].toFixed(2) : '';
      const grad = grads[n.id] !== undefined ? grads[n.id].toFixed(2) : '';
      html += `<rect x="${n.x-40}" y="${n.y-20}" width="100" height="40" fill="#fff" stroke="#6b7280" rx="6"/>`;
      html += `<text x="${n.x+10}" y="${n.y-4}" font-size="11" text-anchor="middle" fill="#111" font-family="ui-monospace,monospace">${n.label}</text>`;
      if (val) html += `<text x="${n.x+10}" y="${n.y+12}" font-size="10" text-anchor="middle" fill="#2563eb" font-family="ui-monospace,monospace">${val}</text>`;
      if (grad) html += `<text x="${n.x+60}" y="${n.y+30}" font-size="10" text-anchor="middle" fill="#dc2626" font-family="ui-monospace,monospace">∂L/∂${n.id}=${grad}</text>`;
    });
    container.querySelector('#bg-svg').innerHTML = html;
    if (phase === 'forward') container.querySelector('#bg-out').textContent = `Forward: loss = ${values.loss.toFixed(3)}`;
    else if (phase === 'backward') container.querySelector('#bg-out').textContent = `Backward: gradients flowed from loss back to x. Now optimizer can update W1, b1, W2, b2 using their gradients.`;
    else container.querySelector('#bg-out').textContent = 'Adjust x and target. Press Forward, then Backward.';
  }

  container.querySelector('#bg-x').addEventListener('input', e => { container.querySelector('#bg-x-out').textContent = e.target.value; });
  container.querySelector('#bg-y').addEventListener('input', e => { container.querySelector('#bg-y-out').textContent = e.target.value; });
  container.querySelector('#bg-fwd').addEventListener('click', forward);
  container.querySelector('#bg-bwd').addEventListener('click', backward);
  container.querySelector('#bg-reset').addEventListener('click', () => { values = {}; grads = {}; phase = 'none'; render(); });
  render();
});
