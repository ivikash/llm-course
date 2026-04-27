// viz/agent_loop.js
// Simulate a ReAct agent loop with buttons to step through.

registerViz('agent_loop', function (container) {
  container.innerHTML = `
    <p class="viz-title">ReAct agent loop</p>
    <p class="viz-sub">A simulated research agent answers "Who wrote the Odyssey?" Step through one iteration at a time.</p>
    <div class="viz-controls">
      <button id="al-step">Next step</button>
      <button id="al-reset">Reset</button>
    </div>
    <div id="al-trace" style="font-family:ui-monospace,monospace;font-size:13px;background:#fafafa;border:1px solid #e5e7eb;border-radius:4px;padding:12px;min-height:200px;white-space:pre-wrap;line-height:1.6"></div>
    <p class="viz-readout" id="al-out"></p>
  `;
  const trace = [
    { type: 'user',    text: 'User: Who wrote the Odyssey?' },
    { type: 'thought', text: 'Thought: I need to look up the author of the Odyssey. I will search the web.' },
    { type: 'action',  text: 'Action: search_web(query="Odyssey author")' },
    { type: 'obs',     text: 'Observation: [{"title":"Odyssey","snippet":"Ancient Greek epic attributed to Homer..."}, {...}]' },
    { type: 'thought', text: 'Thought: The results say Homer. Let me verify with another search to check for "attribution" vs "authorship".' },
    { type: 'action',  text: 'Action: search_web(query="Homer authorship Odyssey scholarly consensus")' },
    { type: 'obs',     text: 'Observation: [{"title":"Homer","snippet":"The Iliad and Odyssey are traditionally attributed to Homer, though the Homeric Question raises doubts..."}]' },
    { type: 'thought', text: 'Thought: Good — traditional attribution is Homer, with some academic debate. I have enough to answer.' },
    { type: 'final',   text: 'Final Answer: The Odyssey is traditionally attributed to Homer, an ancient Greek poet. Modern scholarship (the "Homeric Question") debates whether Homer was a single person or a collection of bards, but Homer remains the conventional author.' },
  ];
  let step = 0;

  function render() {
    const traceEl = container.querySelector('#al-trace');
    let html = '';
    for (let i = 0; i < step; i++) {
      const t = trace[i];
      const colors = {
        user: '#111', thought: '#2563eb', action: '#f59e0b',
        obs: '#059669', final: '#dc2626'
      };
      const icons = { user: '👤', thought: '💭', action: '🔧', obs: '👀', final: '✅' };
      html += `<div style="color:${colors[t.type]};margin-bottom:6px"><b>${icons[t.type]} ${t.text}</b></div>`;
    }
    if (step < trace.length) {
      html += `<div style="color:#9ca3af;margin-top:8px">[press Next step]</div>`;
    } else {
      html += `<div style="color:#15803d;margin-top:8px"><b>Agent loop complete.</b> Total: ${trace.filter(t => t.type==='thought').length} thoughts, ${trace.filter(t => t.type==='action').length} tool calls, 1 final answer.</div>`;
    }
    traceEl.innerHTML = html;
    container.querySelector('#al-out').textContent = `Step ${step}/${trace.length}. ` +
      'Every agent iteration has this shape: Thought → Action → Observation → Thought → ... → Final Answer.';
  }
  container.querySelector('#al-step').addEventListener('click', () => { if (step < trace.length) step++; render(); });
  container.querySelector('#al-reset').addEventListener('click', () => { step = 0; render(); });
  render();
});
