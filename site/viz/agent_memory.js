// viz/agent_memory.js
// Short-term (context window) vs long-term (vector DB) memory dance.

registerViz('agent_memory', function (container) {
  container.innerHTML = `
    <p class="viz-title">Agent memory: short-term vs long-term</p>
    <p class="viz-sub">Send messages. Watch context fill, old ones overflow to long-term memory, and relevant ones get retrieved on demand.</p>
    <div class="viz-controls">
      <label>Context size: <input id="am-c" type="range" min="3" max="10" value="5"><span id="am-c-out">5</span></label>
      <button id="am-add">Send "What's my favorite color?"</button>
      <button id="am-seed">Add 8 sample messages</button>
      <button id="am-reset">Reset</button>
    </div>
    <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:12px">
      <div style="flex:1;min-width:260px">
        <div style="font-weight:bold;font-size:12px;color:#374151;margin-bottom:4px">Short-term (context window)</div>
        <div id="am-ctx" style="background:#dbeafe;border:1px solid #93c5fd;border-radius:4px;padding:8px;min-height:160px;font-size:11px;font-family:ui-monospace,monospace;line-height:1.6"></div>
      </div>
      <div style="flex:1;min-width:260px">
        <div style="font-weight:bold;font-size:12px;color:#374151;margin-bottom:4px">Long-term (vector DB)</div>
        <div id="am-ltm" style="background:#f3f4f6;border:1px solid #d1d5db;border-radius:4px;padding:8px;min-height:160px;font-size:11px;font-family:ui-monospace,monospace;line-height:1.6"></div>
      </div>
    </div>
    <p class="viz-readout" id="am-out">Click "Seed" then "Send". See how old messages move to long-term; the retrieval query brings one back.</p>
  `;

  const seedMsgs = [
    '"I live in Paris."',
    '"My favorite color is blue."',
    '"I have a cat named Luna."',
    '"I work as a software engineer."',
    '"I speak French and English."',
    '"I usually wake up at 7am."',
    '"I love playing chess."',
    '"My birthday is June 15."',
  ];

  let shortTerm = [];
  let longTerm = [];

  function render(highlight) {
    const C = +container.querySelector('#am-c').value;
    container.querySelector('#am-c-out').textContent = C;
    // overflow short to long
    while (shortTerm.length > C) {
      longTerm.push(shortTerm.shift());
    }
    container.querySelector('#am-ctx').innerHTML = shortTerm.length
      ? shortTerm.map((m, i) => `<div style="padding:2px 4px">${i+1}. ${m}</div>`).join('')
      : '<em style="color:#9ca3af">empty</em>';
    container.querySelector('#am-ltm').innerHTML = longTerm.length
      ? longTerm.map((m, i) => {
          const hl = highlight && m === highlight;
          return `<div style="padding:2px 4px;background:${hl ? '#fef3c7' : 'transparent'};border-radius:2px">#${i+1}. ${m}${hl ? ' ← retrieved' : ''}</div>`;
        }).join('')
      : '<em style="color:#9ca3af">empty</em>';
  }

  function addSeed() {
    seedMsgs.forEach(m => shortTerm.push(m));
    render();
    container.querySelector('#am-out').innerHTML = `Added 8 messages. Context holds only the last ${container.querySelector('#am-c').value}; older ones moved to long-term.`;
  }

  function sendQuery() {
    const q = '"What\'s my favorite color?"';
    shortTerm.push(q);
    // retrieve from long-term: simulate by finding "color" in any long-term msg
    const retrieved = longTerm.find(m => /color/i.test(m));
    render(retrieved);
    if (retrieved) {
      container.querySelector('#am-out').innerHTML =
        `Query: ${q}. Cosine similarity search found <b>${retrieved}</b> in long-term. It's injected into the prompt alongside recent context → model answers correctly.`;
    } else {
      container.querySelector('#am-out').innerHTML =
        `Query: ${q}. No relevant long-term memory. Add more seeds first.`;
    }
  }

  function reset() {
    shortTerm = []; longTerm = [];
    render();
  }

  container.querySelector('#am-c').addEventListener('input', () => render());
  container.querySelector('#am-add').addEventListener('click', sendQuery);
  container.querySelector('#am-seed').addEventListener('click', addSeed);
  container.querySelector('#am-reset').addEventListener('click', reset);
  render();
});
