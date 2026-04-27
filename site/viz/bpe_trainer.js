// viz/bpe_trainer.js
// Animated BPE training: corpus shown as tokens, most-frequent-pair merges
// happen step by step.

registerViz('bpe_trainer', function (container) {
  container.innerHTML = `
    <p class="viz-title">BPE tokenizer training, animated</p>
    <p class="viz-sub">Watch Byte-Pair Encoding build its vocabulary. Each step finds the most frequent pair of tokens and merges them.</p>
    <div class="viz-controls">
      <button id="bp-step">Next merge</button>
      <button id="bp-run">Auto-run</button>
      <button id="bp-reset">Reset</button>
    </div>
    <div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:12px">
      <div style="flex:2;min-width:340px">
        <div style="font-size:12px;color:#6b7280;margin-bottom:4px">Corpus (current tokenization)</div>
        <div id="bp-corpus" style="padding:8px;background:#fafafa;border:1px solid #e5e7eb;border-radius:4px;line-height:2;min-height:80px;font-family:ui-monospace,monospace;font-size:12px"></div>
      </div>
      <div style="flex:1;min-width:220px">
        <div style="font-size:12px;color:#6b7280;margin-bottom:4px">Learned vocabulary</div>
        <div id="bp-vocab" style="padding:8px;background:#fafafa;border:1px solid #e5e7eb;border-radius:4px;max-height:200px;overflow:auto;font-family:ui-monospace,monospace;font-size:11px"></div>
      </div>
    </div>
    <p class="viz-readout" id="bp-out">Initial vocab: one token per character.</p>
  `;

  const rawText = 'low low low low low lower lower widest widest widest newest newest newest newest newest newest';
  let tokens = rawText.split(' ').map(w => [...w]); // array of arrays (chars per word)
  let merges = [];
  let baseVocab = new Set();
  rawText.replace(/ /g, '').split('').forEach(c => baseVocab.add(c));

  function findTopPair() {
    const counts = {};
    for (const word of tokens) {
      for (let i = 0; i < word.length - 1; i++) {
        const pair = word[i] + '\x00' + word[i+1];
        counts[pair] = (counts[pair] || 0) + 1;
      }
    }
    let best = null, bestN = 0;
    for (const [k, v] of Object.entries(counts)) {
      if (v > bestN) { bestN = v; best = k; }
    }
    return best ? { pair: best.split('\x00'), count: bestN } : null;
  }

  function merge(pair) {
    tokens = tokens.map(word => {
      const out = [];
      let i = 0;
      while (i < word.length) {
        if (i < word.length - 1 && word[i] === pair[0] && word[i+1] === pair[1]) {
          out.push(pair[0] + pair[1]);
          i += 2;
        } else {
          out.push(word[i]);
          i++;
        }
      }
      return out;
    });
  }

  function render(highlightPair) {
    const corpus = container.querySelector('#bp-corpus');
    corpus.innerHTML = tokens.map(word => {
      const inner = word.map(tok => {
        const isMerge = highlightPair && tok === (highlightPair[0] + highlightPair[1]);
        const bg = isMerge ? '#fde68a' : (tok.length > 1 ? '#dbeafe' : '#fff');
        return `<span style="background:${bg};border:1px solid #d1d5db;border-radius:3px;padding:1px 4px;margin:1px">${tok.replace(/ /g, '·')}</span>`;
      }).join('');
      return `<span style="display:inline-block;margin-right:10px">${inner}</span>`;
    }).join('');
    const vocab = container.querySelector('#bp-vocab');
    const learned = merges.map((m, i) => `<div><span style="color:#6b7280">merge ${i+1}:</span> '${m[0]}' + '${m[1]}' → '<b>${m[0]+m[1]}</b>'</div>`).join('');
    vocab.innerHTML = `<div>base: ${baseVocab.size} chars</div>${learned}`;
  }

  function step() {
    const top = findTopPair();
    if (!top || top.count < 2) {
      container.querySelector('#bp-out').textContent = 'No more frequent pairs. Done.';
      return false;
    }
    merges.push(top.pair);
    merge(top.pair);
    render(top.pair);
    container.querySelector('#bp-out').innerHTML =
      `Merge ${merges.length}: most frequent pair '<b>${top.pair[0]}</b>' + '<b>${top.pair[1]}</b>' (seen <b>${top.count}</b> times) → new token '<b>${top.pair.join('')}</b>'. ` +
      `Vocab size now ${baseVocab.size + merges.length}.`;
    return true;
  }

  let running = null;
  container.querySelector('#bp-step').addEventListener('click', step);
  container.querySelector('#bp-run').addEventListener('click', () => {
    if (running) { clearInterval(running); running = null; return; }
    running = setInterval(() => { if (!step()) { clearInterval(running); running = null; } }, 600);
  });
  container.querySelector('#bp-reset').addEventListener('click', () => {
    if (running) { clearInterval(running); running = null; }
    tokens = rawText.split(' ').map(w => [...w]);
    merges = [];
    render();
    container.querySelector('#bp-out').textContent = 'Reset. Press "Next merge" or "Auto-run".';
  });
  render();
});
