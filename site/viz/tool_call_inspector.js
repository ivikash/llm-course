// viz/tool_call_inspector.js
// Shows an LLM tool call JSON with annotated parts; user picks from examples.

registerViz('tool_call_inspector', function (container) {
  container.innerHTML = `
    <p class="viz-title">Anatomy of a tool call</p>
    <p class="viz-sub">This is what the LLM returns when it decides to call a tool. Hover any piece.</p>
    <div class="viz-controls">
      <label>Example:
        <select id="tc-ex">
          <option value="weather" selected>Get weather</option>
          <option value="search">Web search</option>
          <option value="sql">SQL query</option>
          <option value="parallel">Parallel calls (2 tools at once)</option>
        </select>
      </label>
    </div>
    <pre id="tc-json" style="background:#1e293b;color:#e2e8f0;padding:16px;border-radius:6px;font-size:13px;overflow:auto;line-height:1.6;margin-top:12px"></pre>
    <p class="viz-readout" id="tc-out">Hover any highlighted piece.</p>
  `;

  const examples = {
    weather: {
      json: `{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": "{\\"city\\": \\"Paris\\", \\"unit\\": \\"celsius\\"}"
      }
    }
  ]
}`,
      annotations: [
        { pattern: '"role": "assistant"', note: 'Message is from the LLM.' },
        { pattern: '"content": null', note: 'No text response — the model chose to call a tool instead.' },
        { pattern: '"tool_calls":', note: 'Array of tool invocations. LLMs can call multiple tools in one turn.' },
        { pattern: '"id": "call_abc123"', note: 'Unique ID. You reference this when sending the tool output back.' },
        { pattern: '"name": "get_weather"', note: 'The tool the model wants to invoke.' },
        { pattern: '"arguments":', note: 'Arguments encoded as a JSON STRING (not an object). You json.loads it.' },
      ]
    },
    search: {
      json: `{
  "role": "assistant",
  "content": null,
  "tool_calls": [{
    "id": "call_xyz",
    "type": "function",
    "function": {
      "name": "search_web",
      "arguments": "{\\"query\\": \\"llama 3 release date\\"}"
    }
  }]
}`,
      annotations: []
    },
    sql: {
      json: `{
  "role": "assistant",
  "content": null,
  "tool_calls": [{
    "id": "call_q1",
    "type": "function",
    "function": {
      "name": "run_sql",
      "arguments": "{\\"query\\": \\"SELECT name, COUNT(*) FROM users GROUP BY name LIMIT 5\\"}"
    }
  }]
}`,
      annotations: []
    },
    parallel: {
      json: `{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_w",
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": "{\\"city\\": \\"Paris\\"}"
      }
    },
    {
      "id": "call_c",
      "type": "function",
      "function": {
        "name": "calculator",
        "arguments": "{\\"expression\\": \\"237 * 491\\"}"
      }
    }
  ]
}`,
      annotations: [
        { pattern: '"tool_calls":', note: 'Two calls in parallel — modern models can request multiple tools at once.' }
      ]
    }
  };

  function escape(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

  function render() {
    const key = container.querySelector('#tc-ex').value;
    const ex = examples[key];
    let json = escape(ex.json);
    // annotate hover targets
    const targets = (ex.annotations.length ? ex.annotations : examples.weather.annotations);
    for (const a of targets) {
      const pat = escape(a.pattern);
      json = json.split(pat).join(`<span class="tc-hi" data-note="${encodeURIComponent(a.note)}" style="background:rgba(252,211,77,0.3);cursor:help;padding:1px 2px;border-radius:2px">${pat}</span>`);
    }
    container.querySelector('#tc-json').innerHTML = json;
    container.querySelectorAll('.tc-hi').forEach(el => {
      el.addEventListener('mouseenter', () => {
        container.querySelector('#tc-out').textContent = decodeURIComponent(el.dataset.note);
      });
    });
  }
  container.querySelector('#tc-ex').addEventListener('change', render);
  render();
});
