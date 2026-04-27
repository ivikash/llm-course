// viz/browser_agent.js
// Simulated browser-use agent: fake webpage, agent clicks through.

registerViz('browser_agent', function (container) {
  container.innerHTML = `
    <p class="viz-title">Browser-use agent (simulated)</p>
    <p class="viz-sub">Agent's task: "Find the top story on Hacker News and tell me its score." Press Play; watch it navigate.</p>
    <div class="viz-controls">
      <button id="ba-play">▶ Run agent</button>
      <button id="ba-reset">Reset</button>
    </div>
    <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:12px">
      <div style="flex:2;min-width:340px">
        <div style="background:#111;color:#fff;padding:4px 8px;border-radius:6px 6px 0 0;font-size:11px;font-family:ui-monospace,monospace">
          <span id="ba-url">about:blank</span>
        </div>
        <div id="ba-page" style="border:1px solid #d1d5db;border-top:0;border-radius:0 0 6px 6px;min-height:280px;padding:12px;font-size:13px;background:#fff"></div>
      </div>
      <div style="flex:1;min-width:240px">
        <div style="font-weight:bold;font-size:12px;color:#6b7280;margin-bottom:4px">Agent trace</div>
        <div id="ba-trace" style="padding:8px;background:#fafafa;border:1px solid #e5e7eb;border-radius:4px;font-family:ui-monospace,monospace;font-size:11px;max-height:280px;overflow:auto;white-space:pre-wrap;line-height:1.6"></div>
      </div>
    </div>
    <p class="viz-readout" id="ba-out"></p>
  `;
  const steps = [
    { action: 'navigate',
      args: 'https://news.ycombinator.com',
      thought: 'I need to find Hacker News first.',
      url: 'https://news.ycombinator.com',
      page: `<h2 style="margin-top:0">Hacker News</h2>
            <ol id="hn-list" style="padding-left:24px">
              <li><a href="#" data-id="1">Show HN: Tiny web-based interactive LLM course</a> <span style="color:#6b7280">420 points</span></li>
              <li><a href="#" data-id="2">Discovery: scaling laws for data</a> <span style="color:#6b7280">312 points</span></li>
              <li><a href="#" data-id="3">Rust 1.82 released</a> <span style="color:#6b7280">289 points</span></li>
              <li><a href="#" data-id="4">Why is the sky blue?</a> <span style="color:#6b7280">134 points</span></li>
            </ol>` },
    { action: 'get_content',
      args: '',
      thought: 'Let me read the page to find the top story.',
      url: 'https://news.ycombinator.com',
      page: null },
    { action: 'click',
      args: 'li:first-child a',
      thought: 'The top story is the first <li>. Let me click it to verify the score.',
      url: 'https://news.ycombinator.com/item?id=1',
      page: `<h2 style="margin-top:0">Show HN: Tiny web-based interactive LLM course</h2>
            <p style="color:#6b7280">Posted 3 hours ago by <b>your-handle</b></p>
            <p style="font-size:18px"><b>420 points</b> • 87 comments</p>
            <hr>
            <p>We built an interactive course for learning LLMs from scratch...</p>` },
    { action: 'final',
      thought: 'I have the info I need.',
      final: 'The top story on Hacker News is "Show HN: Tiny web-based interactive LLM course" with 420 points.' },
  ];
  let cur = -1;
  async function play() {
    cur = -1;
    container.querySelector('#ba-trace').textContent = '';
    container.querySelector('#ba-page').innerHTML = '<em style="color:#9ca3af">(blank)</em>';
    container.querySelector('#ba-url').textContent = 'about:blank';
    container.querySelector('#ba-out').textContent = '';
    for (let i = 0; i < steps.length; i++) {
      cur = i;
      const s = steps[i];
      // append to trace
      const trace = container.querySelector('#ba-trace');
      trace.textContent += `\nThought: ${s.thought}`;
      if (s.action === 'final') {
        trace.textContent += `\nFinal Answer: ${s.final}`;
        container.querySelector('#ba-out').innerHTML = `<b>Task complete.</b> Agent used ${steps.length - 1} tool calls over ${steps.length} iterations.`;
      } else {
        trace.textContent += `\nAction: ${s.action}(${s.args})`;
      }
      trace.scrollTop = trace.scrollHeight;
      await new Promise(r => setTimeout(r, 700));
      if (s.url) container.querySelector('#ba-url').textContent = s.url;
      if (s.page) container.querySelector('#ba-page').innerHTML = s.page;
      if (s.action === 'get_content') {
        trace.textContent += '\nObservation: [the full page text]';
      } else if (s.action === 'navigate') {
        trace.textContent += `\nObservation: loaded ${s.url}`;
      } else if (s.action === 'click') {
        trace.textContent += `\nObservation: navigated to ${s.url}`;
      }
      await new Promise(r => setTimeout(r, 500));
    }
  }
  container.querySelector('#ba-play').addEventListener('click', play);
  container.querySelector('#ba-reset').addEventListener('click', () => {
    container.querySelector('#ba-trace').textContent = '';
    container.querySelector('#ba-page').innerHTML = '';
    container.querySelector('#ba-url').textContent = 'about:blank';
    container.querySelector('#ba-out').textContent = '';
  });
});
