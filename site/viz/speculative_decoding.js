// viz/speculative_decoding.js
// Animated draft model -> big model verify. Shows accepted vs rejected tokens.

registerViz('speculative_decoding', function (container) {
  container.innerHTML = `
    <p class="viz-title">Speculative decoding</p>
    <p class="viz-sub">Small "draft" model proposes K tokens cheaply. Big model verifies them all in one forward pass. Accepted ones are kept; first rejection = fallback.</p>
    <div class="viz-controls">
      <label>Draft length K: <input id="sd-k" type="range" min="2" max="8" value="5"><span id="sd-k-out">5</span></label>
      <label>Acceptance rate: <input id="sd-a" type="range" min="0" max="100" value="70"><span id="sd-a-out">70%</span></label>
      <button id="sd-play">▶ Run 3 steps</button>
    </div>
    <div id="sd-trace" style="display:flex;flex-direction:column;gap:8px;margin-top:12px;font-family:ui-monospace,monospace;font-size:12px"></div>
    <p class="viz-readout" id="sd-out"></p>
  `;

  async function play() {
    const K = +container.querySelector('#sd-k').value;
    const accRate = +container.querySelector('#sd-a').value / 100;
    const trace = container.querySelector('#sd-trace');
    trace.innerHTML = '';

    const words = ['The', 'cat', 'sat', 'on', 'the', 'mat', 'and', 'yawned', 'in', 'the', 'warm', 'sun', 'today'];
    let pos = 0;
    let totalGen = 0;
    let totalBigCalls = 0;

    for (let step = 0; step < 3; step++) {
      // Draft model proposes K tokens
      const drafts = words.slice(pos, pos + K);
      const row1 = document.createElement('div');
      row1.innerHTML = `<span style="color:#6b7280">step ${step+1}:</span> draft proposes: `;
      drafts.forEach((w, i) => {
        const s = document.createElement('span');
        s.textContent = w;
        s.style.cssText = 'background:#fef3c7;color:#78350f;padding:2px 6px;margin:0 2px;border-radius:3px';
        row1.appendChild(s);
      });
      trace.appendChild(row1);
      await new Promise(r => setTimeout(r, 500));

      // Big model verifies: how many got accepted?
      let accepted = 0;
      for (let i = 0; i < K; i++) {
        if (Math.random() < accRate) accepted++;
        else break;
      }
      const row2 = document.createElement('div');
      row2.innerHTML = `  big model verifies: `;
      drafts.forEach((w, i) => {
        const s = document.createElement('span');
        s.textContent = w;
        if (i < accepted) {
          s.style.cssText = 'background:#bbf7d0;color:#14532d;padding:2px 6px;margin:0 2px;border-radius:3px';
          s.textContent = '✓ ' + w;
        } else if (i === accepted) {
          s.style.cssText = 'background:#fecaca;color:#7f1d1d;padding:2px 6px;margin:0 2px;border-radius:3px';
          s.textContent = '✗ ' + w;
        } else {
          s.style.cssText = 'background:#e5e7eb;color:#9ca3af;padding:2px 6px;margin:0 2px;border-radius:3px;text-decoration:line-through';
        }
        row2.appendChild(s);
      });
      // +1 correction token from big model when rejection happens
      if (accepted < K) {
        const fb = document.createElement('span');
        fb.textContent = '+correction';
        fb.style.cssText = 'background:#2563eb;color:#fff;padding:2px 6px;margin:0 2px;border-radius:3px';
        row2.appendChild(fb);
      }
      trace.appendChild(row2);
      totalGen += accepted + (accepted < K ? 1 : 0);
      totalBigCalls += 1;
      pos += accepted + (accepted < K ? 1 : 0);
      await new Promise(r => setTimeout(r, 500));
    }

    const speedup = totalGen / totalBigCalls;
    container.querySelector('#sd-out').innerHTML =
      `Generated <b>${totalGen}</b> tokens with only <b>${totalBigCalls}</b> big-model forward passes. ` +
      `Effective speedup: <b>${speedup.toFixed(2)}×</b>. ` +
      `Higher acceptance rate (good draft model) → higher speedup. Acceptance rate 100% would give ~${+container.querySelector('#sd-k').value}× speedup.`;
  }
  container.querySelector('#sd-k').addEventListener('input', e => { container.querySelector('#sd-k-out').textContent = e.target.value; });
  container.querySelector('#sd-a').addEventListener('input', e => { container.querySelector('#sd-a-out').textContent = e.target.value + '%'; });
  container.querySelector('#sd-play').addEventListener('click', play);
  play();
});
