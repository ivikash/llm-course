// viz/grpo_rollouts.js
// Simulates GRPO: sample K responses, get 0/1 reward, compute advantage.

registerViz('grpo_rollouts', function (container) {
  container.innerHTML = `
    <p class="viz-title">GRPO rollouts: learning from correctness</p>
    <p class="viz-sub">Sample K responses to a math problem. Score each correct/wrong. Advantage = (reward − group_mean) / std. Right answers get pushed UP, wrong answers DOWN.</p>
    <div class="viz-controls">
      <label>K (rollouts): <input id="gr-k" type="range" min="4" max="32" value="16"><span id="gr-k-out">16</span></label>
      <label>Model skill: <input id="gr-s" type="range" min="0" max="100" value="40"><span id="gr-s-out">40%</span></label>
      <button id="gr-sample">Sample again</button>
    </div>
    <div id="gr-grid" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:6px;margin-top:12px"></div>
    <p class="viz-readout" id="gr-out"></p>
  `;
  function sample() {
    const K = +container.querySelector('#gr-k').value;
    const skill = +container.querySelector('#gr-s').value / 100;
    container.querySelector('#gr-k-out').textContent = K;
    container.querySelector('#gr-s-out').textContent = (skill * 100).toFixed(0) + '%';
    const rewards = [];
    for (let i = 0; i < K; i++) rewards.push(Math.random() < skill ? 1 : 0);
    const mean = rewards.reduce((a, b) => a + b, 0) / K;
    const std = Math.sqrt(rewards.reduce((s, r) => s + (r - mean) ** 2, 0) / K) || 0.01;
    const advantages = rewards.map(r => (r - mean) / std);

    let html = '';
    const texts = [
      'Answer: 12', 'Answer: 15', 'Answer: 12', 'Let me compute...', 'I think 12', '12', 'The result is 12',
      'Hmm, 15 maybe?', '11', '12 I believe', 'Not sure but 12', 'Final: 12', '12?', '13', '12', 'Seems like 12'
    ];
    for (let i = 0; i < K; i++) {
      const r = rewards[i], a = advantages[i];
      const color = r === 1 ? '#bbf7d0' : '#fecaca';
      const border = a > 0 ? '#16a34a' : a < 0 ? '#dc2626' : '#9ca3af';
      html += `<div style="padding:8px;border:2px solid ${border};border-radius:6px;background:${color};font-size:11px;text-align:center">
        <div style="font-family:ui-monospace,monospace">${texts[i % texts.length]}</div>
        <div style="margin-top:4px;font-weight:bold">r=${r}, A=${a.toFixed(2)}</div>
      </div>`;
    }
    container.querySelector('#gr-grid').innerHTML = html;
    const numRight = rewards.filter(r => r === 1).length;
    container.querySelector('#gr-out').innerHTML =
      `<b>${numRight}/${K}</b> correct (${(mean * 100).toFixed(0)}%). ` +
      `<span style="color:#16a34a">Right answers get advantage +${((1-mean)/std).toFixed(2)}</span> → policy updated to make them MORE likely. ` +
      `<span style="color:#dc2626">Wrong answers get advantage ${((-mean)/std).toFixed(2)}</span> → LESS likely. ` +
      `Model learns without human labels.`;
  }
  container.querySelector('#gr-sample').addEventListener('click', sample);
  container.querySelector('#gr-k').addEventListener('input', sample);
  container.querySelector('#gr-s').addEventListener('input', sample);
  sample();
});
