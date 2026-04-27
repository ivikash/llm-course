// viz/reasoning_trace.js
// Compare a non-reasoning model's single-shot answer to an o1-style
// model's chain-of-thought (with backtracking).

registerViz('reasoning_trace', function (container) {
  container.innerHTML = `
    <p class="viz-title">Reasoning models: show your work</p>
    <p class="viz-sub">Same math problem. Left: direct answer (often wrong). Right: o1/R1-style extended reasoning (correct, with backtracking).</p>
    <div class="viz-controls">
      <button id="rt-run">Run both</button>
    </div>
    <div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:12px">
      <div style="flex:1;min-width:300px;border:1px solid #e5e7eb;border-radius:6px;padding:12px">
        <div style="font-weight:bold;color:#6b7280;margin-bottom:8px">Regular model (GPT-4o-style)</div>
        <div id="rt-regular" style="font-family:ui-monospace,monospace;font-size:12px;background:#fafafa;padding:8px;border-radius:4px;min-height:200px;white-space:pre-wrap"></div>
      </div>
      <div style="flex:1;min-width:300px;border:1px solid #e5e7eb;border-radius:6px;padding:12px">
        <div style="font-weight:bold;color:#10b981;margin-bottom:8px">Reasoning model (o1/R1-style)</div>
        <div id="rt-reason" style="font-family:ui-monospace,monospace;font-size:12px;background:#fafafa;padding:8px;border-radius:4px;min-height:200px;white-space:pre-wrap"></div>
      </div>
    </div>
    <p class="viz-readout" id="rt-out"></p>
  `;
  const question = 'Problem: Alice has twice as many apples as Bob. Charlie has 3 more than Alice. Together they have 23 apples. How many does Bob have?';
  const regular = [
    'Let me set up variables and solve.',
    'Bob has B apples. Alice has 2B. Charlie has 2B+3.',
    'Total: B + 2B + 2B+3 = 23',
    '5B + 3 = 23',
    '5B = 20',
    'B = 5',
    '',
    'Answer: Bob has 5 apples.',
  ];
  const reasoning = [
    'Let me set up variables carefully.',
    '',
    'Let B = number of apples Bob has.',
    'Then Alice has 2B (twice as many).',
    'Charlie has 2B + 3 (three more than Alice).',
    '',
    'Sum: B + 2B + (2B + 3) = 23',
    '5B + 3 = 23',
    '5B = 20',
    'B = 4. Hmm wait, let me double-check.',
    '',
    'Actually 20/5 = 4. So B = 4.',
    '',
    'Verify: Bob=4, Alice=8, Charlie=11. Sum=4+8+11=23. ✓',
    '',
    'Wait — I initially wrote 5. Let me redo:',
    '5B = 20 → B = 4. Yes, 4 is correct.',
    '',
    'Final answer: Bob has 4 apples.',
  ];
  async function stream(targetId, lines, delay) {
    const el = container.querySelector('#' + targetId);
    el.textContent = '';
    for (const line of lines) {
      el.textContent += line + '\n';
      await new Promise(r => setTimeout(r, delay));
    }
  }
  async function run() {
    container.querySelector('#rt-regular').textContent = question + '\n\n';
    container.querySelector('#rt-reason').textContent = question + '\n\n';
    await new Promise(r => setTimeout(r, 500));
    const regP = stream('rt-regular', regular, 120);
    const reaP = stream('rt-reason', reasoning, 120);
    await Promise.all([regP, reaP]);
    container.querySelector('#rt-out').innerHTML =
      `<b style="color:#dc2626">Regular: answered 5 — wrong</b>. ` +
      `<b style="color:#10b981">Reasoning: answered 4 — correct, and caught its own mistake.</b> ` +
      'Reasoning models spend 5-100× more tokens thinking but get hard problems right. Cost vs accuracy trade-off.';
  }
  container.querySelector('#rt-run').addEventListener('click', run);
  run();
});
