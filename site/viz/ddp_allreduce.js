// viz/ddp_allreduce.js
// Animate 8-GPU DDP step: forward/backward on each, then all-reduce
// gradients, then optimizer step.

registerViz('ddp_allreduce', function (container) {
  container.innerHTML = `
    <p class="viz-title">DDP: 8 GPUs training together</p>
    <p class="viz-sub">Each GPU has a full model copy, processes a different batch, then all GPUs sync gradients via all-reduce. Press Play to see one training step.</p>
    <div class="viz-controls">
      <label>GPUs: <input id="ddp-n" type="range" min="2" max="16" value="8"><span id="ddp-n-out">8</span></label>
      <button id="ddp-play">▶ Play one step</button>
    </div>
    <svg id="ddp-svg" viewBox="0 0 720 320" width="720" height="320" style="background:#fff;border:1px solid #e5e7eb;border-radius:4px;max-width:100%"></svg>
    <p class="viz-readout" id="ddp-out"></p>
  `;
  const svg = container.querySelector('#ddp-svg');
  const out = container.querySelector('#ddp-out');
  const nEl = container.querySelector('#ddp-n');
  const phases = [
    { name: 'forward',       desc: 'Each GPU runs forward on its own batch slice.' },
    { name: 'backward',      desc: 'Each GPU computes its own gradient.' },
    { name: 'all-reduce',    desc: 'Gradients summed across all GPUs, divided by N, broadcast back. This is the network-heavy step (NCCL ring all-reduce).' },
    { name: 'optimizer-step', desc: 'Each GPU runs the same optimizer step on the averaged gradients. Weights stay in sync.' },
  ];
  let phase = -1;

  function render() {
    const N = +nEl.value;
    container.querySelector('#ddp-n-out').textContent = N;
    const cols = Math.min(N, 8);
    const rows = Math.ceil(N / cols);
    const cw = 680 / cols, rh = 120;
    let html = '';
    // draw GPUs
    for (let i = 0; i < N; i++) {
      const r = Math.floor(i / cols), c = i % cols;
      const x = 20 + c * cw + 8;
      const y = 40 + r * rh;
      const fwColor = phase >= 0 ? '#60a5fa' : '#e5e7eb';
      const bwColor = phase >= 1 ? '#f59e0b' : '#e5e7eb';
      const stepColor = phase >= 3 ? '#10b981' : '#e5e7eb';
      html += `<rect x="${x}" y="${y}" width="${cw - 16}" height="${rh - 10}" fill="#fff" stroke="#6b7280" rx="6"/>`;
      html += `<text x="${x+8}" y="${y+18}" font-size="11" fill="#111" font-weight="bold">GPU ${i}</text>`;
      html += `<text x="${x+8}" y="${y+34}" font-size="10" fill="#6b7280">batch ${i}</text>`;
      html += `<rect x="${x+8}" y="${y+44}" width="${cw - 32}" height="16" fill="${fwColor}" rx="3"/>`;
      html += `<text x="${x+14}" y="${y+56}" font-size="10" fill="#fff">fwd</text>`;
      html += `<rect x="${x+8}" y="${y+64}" width="${cw - 32}" height="16" fill="${bwColor}" rx="3"/>`;
      html += `<text x="${x+14}" y="${y+76}" font-size="10" fill="#fff">bwd</text>`;
      html += `<rect x="${x+8}" y="${y+84}" width="${cw - 32}" height="16" fill="${stepColor}" rx="3"/>`;
      html += `<text x="${x+14}" y="${y+96}" font-size="10" fill="#fff">step</text>`;
    }
    // draw all-reduce arcs
    if (phase === 2) {
      // draw a ring connecting all GPUs
      for (let i = 0; i < N; i++) {
        const r = Math.floor(i / cols), c = i % cols;
        const x = 20 + c * cw + cw/2 - 8;
        const y = 40 + r * rh + rh/2 - 5;
        const next = (i + 1) % N;
        const rn = Math.floor(next / cols), cn = next % cols;
        const xn = 20 + cn * cw + cw/2 - 8;
        const yn = 40 + rn * rh + rh/2 - 5;
        html += `<line x1="${x}" y1="${y}" x2="${xn}" y2="${yn}" stroke="#dc2626" stroke-width="2" marker-end="url(#ddp-arrow)" opacity="0.6"/>`;
      }
      html = `<defs><marker id="ddp-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="#dc2626"/></marker></defs>` + html;
    }
    svg.innerHTML = html;
    if (phase >= 0) out.textContent = `Phase ${phase+1}/4: ${phases[phase].name}. ${phases[phase].desc}`;
    else out.textContent = 'Press Play to watch one DDP training step.';
  }

  async function play() {
    for (phase = 0; phase < phases.length; phase++) {
      render();
      await new Promise(r => setTimeout(r, 1200));
    }
    phase = -1;
    setTimeout(render, 500);
  }

  nEl.addEventListener('input', render);
  container.querySelector('#ddp-play').addEventListener('click', play);
  render();
});
