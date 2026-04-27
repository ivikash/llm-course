// viz/live_tokenizer.js
// In-browser tokenizer visualizer. Uses a simplified word-boundary
// BPE-ish tokenization for illustration. Shows colored tokens like tiktokenizer.

registerViz('live_tokenizer', function (container) {
  container.innerHTML = `
    <p class="viz-title">Live tokenizer</p>
    <p class="viz-sub">Paste text. See how a BPE-style tokenizer splits it into colored tokens. Each token → one integer ID. Every LLM sees only these integers.</p>
    <textarea id="lt-in" rows="3" style="width:100%;padding:8px;border:1px solid #d1d5db;border-radius:4px;font-family:ui-monospace,monospace">Hello, world! The quick brown fox jumps over the lazy dog.</textarea>
    <div class="viz-controls">
      <label>Mode:
        <select id="lt-mode">
          <option value="word">word-ish (simple)</option>
          <option value="char">character-level</option>
          <option value="bpe" selected>BPE-ish (simulated)</option>
        </select>
      </label>
    </div>
    <div id="lt-tokens" style="line-height:2;padding:8px;background:#fafafa;border-radius:4px;border:1px solid #e5e7eb;min-height:60px"></div>
    <p class="viz-readout" id="lt-stats"></p>
  `;

  // Simulated BPE merges: common English patterns.
  // Not real tiktoken but same idea - reveals that " the", "ing", "ed", etc. become single tokens.
  const bpePatterns = [
    // longest-first; greedy matching
    /^ the\b/, /^ and\b/, /^ of\b/, /^ to\b/, /^ is\b/, /^ in\b/, /^ on\b/, /^ at\b/, /^ for\b/, /^ with\b/,
    /^ that\b/, /^ this\b/, /^ a\b/, /^ an\b/, /^ it\b/, /^ you\b/, /^ are\b/, /^ was\b/, /^ will\b/,
    /^ing\b/, /^ed\b/, /^ly\b/, /^tion\b/, /^ness\b/,
    /^ [A-Z][a-z]+/, /^[A-Z][a-z]+/, /^ [a-z]+/, /^[a-z]+/, /^\d+/, /^[.,!?;:]/, /^\s/, /^./
  ];

  // color palette
  const colors = ['#fee2e2','#fed7aa','#fef3c7','#d9f99d','#a7f3d0','#bae6fd','#c7d2fe','#e9d5ff','#fbcfe8','#fda4af'];

  function tokenize(text, mode) {
    const tokens = [];
    if (mode === 'char') {
      for (const c of text) tokens.push(c);
      return tokens;
    }
    if (mode === 'word') {
      return text.split(/(\s+|[.,!?;:])/).filter(Boolean);
    }
    // bpe-ish: greedy longest-match
    let rest = text;
    while (rest.length > 0) {
      let matched = false;
      for (const pat of bpePatterns) {
        const m = rest.match(pat);
        if (m) {
          tokens.push(m[0]);
          rest = rest.slice(m[0].length);
          matched = true;
          break;
        }
      }
      if (!matched) {
        tokens.push(rest[0]);
        rest = rest.slice(1);
      }
    }
    return tokens;
  }

  function hash(s) { // simple hash → stable integer id
    let h = 0;
    for (const c of s) h = (h * 31 + c.charCodeAt(0)) >>> 0;
    return h % 50000;
  }

  function render() {
    const text = container.querySelector('#lt-in').value;
    const mode = container.querySelector('#lt-mode').value;
    const toks = tokenize(text, mode);
    const outEl = container.querySelector('#lt-tokens');
    outEl.innerHTML = toks.map((t, i) => {
      const c = colors[i % colors.length];
      const id = hash(t);
      const disp = t.replace(/ /g, '␣').replace(/\n/g, '↵');
      return `<span title="id=${id}, length=${t.length}" style="background:${c};padding:2px 6px;border-radius:4px;margin:2px;display:inline-block;font-family:ui-monospace,monospace;font-size:13px">${disp}</span>`;
    }).join('');
    const bytes = new TextEncoder().encode(text).length;
    const ratio = bytes && toks.length ? (bytes / toks.length).toFixed(2) : '-';
    container.querySelector('#lt-stats').textContent =
      `${toks.length} tokens, ${bytes} bytes, compression = ${ratio} bytes/token (higher = better compression, more tokens carry more info)`;
  }

  container.querySelector('#lt-in').addEventListener('input', render);
  container.querySelector('#lt-mode').addEventListener('change', render);
  render();
});
