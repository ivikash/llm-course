// viz/vlm_tokens.js
// Show how a VLM (LLaVA) composes image patches + text tokens into one sequence.

registerViz('vlm_tokens', function (container) {
  container.innerHTML = `
    <p class="viz-title">VLM input: image tokens + text tokens in one sequence</p>
    <p class="viz-sub">LLaVA / GPT-4V convert an image into "visual tokens" (projected CLIP features) that sit in the LLM's context alongside text tokens. Same transformer processes both.</p>
    <div class="viz-controls">
      <label>Image: <select id="vt-img">
        <option value="cat" selected>🐱 cat photo</option>
        <option value="chart">📊 bar chart</option>
        <option value="street">🚦 street scene</option>
      </select></label>
      <label>User question:
        <select id="vt-q">
          <option>What is in this image?</option>
          <option>Describe it in one word.</option>
          <option>Is there anything dangerous?</option>
        </select>
      </label>
    </div>
    <div id="vt-seq" style="display:flex;flex-wrap:wrap;gap:3px;padding:12px;background:#fafafa;border:1px solid #e5e7eb;border-radius:4px;margin-top:12px;line-height:2"></div>
    <p class="viz-readout" id="vt-out"></p>
  `;

  const imgs = {
    cat:    { emoji: '🐱', kind: 'cat' },
    chart:  { emoji: '📊', kind: 'chart' },
    street: { emoji: '🚦', kind: 'street' },
  };

  function render() {
    const imgKey = container.querySelector('#vt-img').value;
    const q = container.querySelector('#vt-q').value;
    const img = imgs[imgKey];
    const visualTokens = 576;  // LLaVA-1.5 default
    const seqEl = container.querySelector('#vt-seq');
    let html = '';
    // image start marker
    html += `<span style="background:#f59e0b;color:#fff;padding:2px 6px;border-radius:3px;font-family:ui-monospace,monospace;font-size:11px">&lt;|image_start|&gt;</span>`;
    // visual tokens (show 24 to represent 576)
    for (let i = 0; i < 24; i++) {
      const hue = (i * 11) % 360;
      html += `<span title="visual token ${i+1} (projected CLIP feature)" style="background:hsl(${hue},70%,85%);padding:2px 6px;border-radius:3px;font-family:ui-monospace,monospace;font-size:11px;cursor:help">v${i+1}</span>`;
    }
    html += `<span style="color:#9ca3af;padding:2px 6px">... (${visualTokens} visual tokens total, 1 per image patch)</span>`;
    html += `<span style="background:#f59e0b;color:#fff;padding:2px 6px;border-radius:3px;font-family:ui-monospace,monospace;font-size:11px">&lt;|image_end|&gt;</span>`;
    // text tokens (user question, tokenized word-ish)
    q.split(/(\s+|[?.])/g).filter(Boolean).forEach(w => {
      if (/\s+/.test(w)) return;
      html += `<span style="background:#dbeafe;color:#1e3a8a;padding:2px 6px;border-radius:3px;font-family:ui-monospace,monospace;font-size:11px">${w.replace(/ /g,'·')}</span>`;
    });
    seqEl.innerHTML = html;

    // stats
    const textTok = q.trim().split(/\s+/).length + 1;
    const total = visualTokens + textTok + 2;
    container.querySelector('#vt-out').innerHTML =
      `Total sequence length: <b>${total}</b> tokens (${visualTokens} visual + ${textTok} text + 2 markers). ` +
      `The LLM treats them identically — attention flows freely between image and text. ` +
      `That's how VLMs can "reason about images" without any special image-handling code in the transformer.`;
  }
  container.querySelector('#vt-img').addEventListener('change', render);
  container.querySelector('#vt-q').addEventListener('change', render);
  render();
});
