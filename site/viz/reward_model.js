// viz/reward_model.js
// Pairwise preference -> reward scores. Pick which response is better;
// watch logits + Bradley-Terry loss respond.

registerViz('reward_model', function (container) {
  container.innerHTML = `
    <p class="viz-title">Reward model training step</p>
    <p class="viz-sub">Human labels 2 model responses: which is better? Model emits scalar rewards; Bradley-Terry loss pushes the chosen one up, rejected down.</p>
    <div style="background:#f3f4f6;border-radius:6px;padding:12px;margin-top:12px">
      <div style="font-size:11px;color:#6b7280;margin-bottom:6px">Prompt</div>
      <div style="font-family:ui-monospace,monospace;font-size:12px">Explain why the sky is blue.</div>
    </div>
    <div style="display:flex;gap:12px;margin-top:12px;flex-wrap:wrap">
      <div id="rm-a" style="flex:1;min-width:240px;border:2px solid #d1d5db;border-radius:6px;padding:12px;cursor:pointer;background:#fff">
        <div style="font-size:11px;color:#6b7280;margin-bottom:4px">Response A</div>
        <div style="font-size:12px;line-height:1.5">Light scatters off air molecules. Shorter blue wavelengths scatter more than red, so that's what we see.</div>
        <div style="margin-top:8px">Reward: <b id="rm-ra" style="font-family:ui-monospace,monospace">0.0</b></div>
      </div>
      <div id="rm-b" style="flex:1;min-width:240px;border:2px solid #d1d5db;border-radius:6px;padding:12px;cursor:pointer;background:#fff">
        <div style="font-size:11px;color:#6b7280;margin-bottom:4px">Response B</div>
        <div style="font-size:12px;line-height:1.5">Because it just is. Don't ask me, I'm just a language model.</div>
        <div style="margin-top:8px">Reward: <b id="rm-rb" style="font-family:ui-monospace,monospace">0.0</b></div>
      </div>
    </div>
    <div class="viz-controls" style="margin-top:12px">
      <button id="rm-pick-a">👍 A is better</button>
      <button id="rm-pick-b">👍 B is better</button>
      <button id="rm-reset">Reset</button>
    </div>
    <p class="viz-readout" id="rm-out">Click which response is better. The model will adjust.</p>
  `;
  let ra = 0, rb = 0;
  let history = [];

  function update() {
    container.querySelector('#rm-ra').textContent = ra.toFixed(2);
    container.querySelector('#rm-rb').textContent = rb.toFixed(2);
    // style based on which is ahead
    const aEl = container.querySelector('#rm-a');
    const bEl = container.querySelector('#rm-b');
    aEl.style.borderColor = ra > rb ? '#10b981' : '#d1d5db';
    bEl.style.borderColor = rb > ra ? '#10b981' : '#d1d5db';
    // loss
    const diff = ra - rb;
    const prob = 1 / (1 + Math.exp(-diff));
    const lossA = -Math.log(prob);
    const lossB = -Math.log(1 - prob);
    const last = history[history.length - 1];
    const loss = last === 'a' ? lossA : last === 'b' ? lossB : 0;
    container.querySelector('#rm-out').innerHTML =
      `Reward gap = <b>${diff.toFixed(2)}</b>. ` +
      `P(A preferred | rewards) = σ(r_A − r_B) = <b>${prob.toFixed(3)}</b>. ` +
      (last ? `Last label was '${last.toUpperCase()} is better'; loss = <b>${loss.toFixed(3)}</b>.` : '') +
      ` ${history.length} preferences trained on.`;
  }

  function step(choice) {
    // simple SGD: push chosen up, rejected down
    const lr = 0.15;
    const diff = ra - rb;
    const pA = 1 / (1 + Math.exp(-diff));
    if (choice === 'a') {
      // grad of -log(pA) wrt ra, rb:
      ra += lr * (1 - pA);
      rb -= lr * (1 - pA);
    } else {
      ra -= lr * pA;
      rb += lr * pA;
    }
    history.push(choice);
    update();
  }

  container.querySelector('#rm-pick-a').addEventListener('click', () => step('a'));
  container.querySelector('#rm-pick-b').addEventListener('click', () => step('b'));
  container.querySelector('#rm-reset').addEventListener('click', () => { ra = 0; rb = 0; history = []; update(); });
  update();
});
