// viz/attention_interactive.js
// The big one: click a token, see what it attends to in a sentence.
// Uses hand-crafted attention weights that illustrate real patterns
// (coreference, local, syntactic) so the viz is educational without
// shipping a real transformer in-browser.

registerViz('attention_interactive', function (container) {
  container.innerHTML = `
    <p class="viz-title">Attention: click a token, see what it looks at</p>
    <p class="viz-sub">A sentence with hand-crafted attention weights that show realistic patterns. Try clicking <b>"it"</b> — you'll see it attend strongly to "animal" (coreference resolution).</p>
    <div class="viz-controls">
      <label>Head:
        <select id="ai-head">
          <option value="coref">Head 1: Coreference (pronouns → referents)</option>
          <option value="local">Head 2: Local (previous token)</option>
          <option value="syntactic">Head 3: Syntactic (verb → subject)</option>
          <option value="content">Head 4: Content (nouns → adjectives)</option>
        </select>
      </label>
    </div>
    <div id="ai-tokens" style="display:flex;flex-wrap:wrap;gap:8px;padding:16px;background:#fafafa;border-radius:6px;min-height:60px"></div>
    <svg id="ai-svg" viewBox="0 0 800 240" width="800" height="240" style="background:#fff;border:1px solid #e5e7eb;border-radius:4px;margin-top:12px;max-width:100%"></svg>
    <p class="viz-readout" id="ai-out">Click any token above to see its attention pattern.</p>
  `;

  const tokens = ['The', 'animal', 'did', 'not', 'cross', 'the', 'street',
                  'because', 'it', 'was', 'too', 'tired', '.'];
  const N = tokens.length;

  // Hand-crafted attention heads (row = query, col = key, probs sum to 1 per row up to index)
  function buildHeads() {
    const heads = {};
    // 1. Coreference: "it" attends to "animal"
    heads.coref = Array.from({ length: N }, (_, i) => {
      const row = new Array(N).fill(0);
      if (tokens[i].toLowerCase() === 'it') {
        row[1] = 0.7;  // animal
        row[4] = 0.08; // cross
        row[6] = 0.08; // street
        row[i] = 0.1;
        for (let j = 0; j <= i; j++) if (row[j] === 0) row[j] = 0.04 / Math.max(1, i - 2);
      } else {
        row[i] = 0.5;
        for (let j = 0; j < i; j++) row[j] = 0.5 / Math.max(1, i);
      }
      const s = row.slice(0, i + 1).reduce((a, b) => a + b);
      for (let j = 0; j <= i; j++) row[j] /= s;
      return row;
    });
    // 2. Local: mostly attends to previous 1-2 tokens
    heads.local = Array.from({ length: N }, (_, i) => {
      const row = new Array(N).fill(0);
      if (i === 0) row[0] = 1;
      else { row[i] = 0.3; row[i - 1] = 0.6; if (i >= 2) row[i - 2] = 0.1; }
      const s = row.slice(0, i + 1).reduce((a, b) => a + b);
      for (let j = 0; j <= i; j++) row[j] /= s;
      return row;
    });
    // 3. Syntactic: verbs attend to their subjects
    heads.syntactic = Array.from({ length: N }, (_, i) => {
      const row = new Array(N).fill(0);
      const tok = tokens[i].toLowerCase();
      if (['did', 'cross', 'was'].includes(tok)) {
        row[1] = 0.5; // animal (subject)
        row[i] = 0.3;
        for (let j = 0; j < i; j++) if (row[j] === 0) row[j] = 0.2 / Math.max(1, i);
      } else {
        row[i] = 0.6;
        for (let j = 0; j < i; j++) row[j] = 0.4 / Math.max(1, i);
      }
      const s = row.slice(0, i + 1).reduce((a, b) => a + b);
      for (let j = 0; j <= i; j++) row[j] /= s;
      return row;
    });
    // 4. Content: "tired" attends to "animal"
    heads.content = Array.from({ length: N }, (_, i) => {
      const row = new Array(N).fill(0);
      if (tokens[i].toLowerCase() === 'tired') {
        row[1] = 0.5; // animal
        row[8] = 0.3; // it
        row[i] = 0.2;
      } else {
        row[i] = 0.7;
        for (let j = 0; j < i; j++) row[j] = 0.3 / Math.max(1, i);
      }
      const s = row.slice(0, i + 1).reduce((a, b) => a + b);
      for (let j = 0; j <= i; j++) row[j] /= s;
      return row;
    });
    return heads;
  }
  const heads = buildHeads();

  const tokensEl = container.querySelector('#ai-tokens');
  const svg = container.querySelector('#ai-svg');
  const out = container.querySelector('#ai-out');
  const headEl = container.querySelector('#ai-head');

  let selectedQuery = 8; // default: "it"

  function render() {
    const head = heads[headEl.value];
    const weights = head[selectedQuery];
    // render tokens with highlighting
    tokensEl.innerHTML = tokens.map((t, i) => {
      const w = weights[i];
      const opacity = i <= selectedQuery ? Math.min(1, w * 2) : 0.1;
      const bg = i === selectedQuery ? '#fbbf24' :
                 (i > selectedQuery) ? '#e5e7eb' :
                 `rgba(37, 99, 235, ${opacity.toFixed(2)})`;
      const color = i <= selectedQuery ? '#111' : '#9ca3af';
      return `<div data-idx="${i}" style="cursor:pointer;padding:8px 12px;background:${bg};color:${color};border:1px solid #d1d5db;border-radius:6px;font-size:14px;font-family:ui-monospace,monospace">${t}<div style="font-size:10px;color:#6b7280">${(w * 100).toFixed(0)}%</div></div>`;
    }).join('');

    // draw attention connections as arcs
    const tokW = 800 / N;
    let arcs = '';
    const queryX = selectedQuery * tokW + tokW / 2;
    for (let k = 0; k <= selectedQuery; k++) {
      const w = weights[k];
      if (w < 0.03) continue;
      const keyX = k * tokW + tokW / 2;
      const midX = (queryX + keyX) / 2;
      const arcHeight = Math.abs(queryX - keyX) * 0.4 + 20;
      arcs += `<path d="M ${queryX} 200 Q ${midX} ${200 - arcHeight} ${keyX} 200"
                     fill="none" stroke="rgba(37,99,235,${w.toFixed(2)})" stroke-width="${1 + w * 4}"/>`;
    }
    let tokLabels = '';
    for (let i = 0; i < N; i++) {
      const x = i * tokW + tokW / 2;
      const color = i === selectedQuery ? '#ea580c' : '#374151';
      const fontWeight = i === selectedQuery ? 'bold' : 'normal';
      tokLabels += `<text x="${x}" y="220" font-size="12" text-anchor="middle" fill="${color}" font-weight="${fontWeight}">${tokens[i]}</text>`;
      tokLabels += `<circle cx="${x}" cy="200" r="3" fill="${color}"/>`;
    }
    svg.innerHTML = arcs + tokLabels;

    out.innerHTML = `Query = <b style="color:#ea580c">"${tokens[selectedQuery]}"</b>. ` +
      `Top attended keys: ` +
      weights.slice(0, selectedQuery + 1)
        .map((w, i) => ({ w, tok: tokens[i], i }))
        .sort((a, b) => b.w - a.w)
        .slice(0, 3)
        .map(x => `"<b>${x.tok}</b>" (${(x.w * 100).toFixed(0)}%)`)
        .join(', ');

    tokensEl.querySelectorAll('[data-idx]').forEach(el => {
      el.addEventListener('click', () => {
        selectedQuery = +el.dataset.idx;
        render();
      });
    });
  }

  headEl.addEventListener('change', render);
  render();
});
