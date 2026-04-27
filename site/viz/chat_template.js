// viz/chat_template.js
// Show how raw conversation gets rendered into the actual token stream
// via a chat template. Compare ChatML / Llama-3 / nanochat formats.

registerViz('chat_template', function (container) {
  container.innerHTML = `
    <p class="viz-title">Chat templates: what the model actually sees</p>
    <p class="viz-sub">Your nice "list of messages" gets flattened into one token stream with special tokens as delimiters. Different models, different templates.</p>
    <div class="viz-controls">
      <label>Template:
        <select id="ct-t">
          <option value="chatml" selected>ChatML (OpenAI, Qwen)</option>
          <option value="llama3">Llama-3</option>
          <option value="nanochat">nanochat</option>
        </select>
      </label>
    </div>
    <div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:12px">
      <div style="flex:1;min-width:260px">
        <div style="font-size:11px;color:#6b7280;margin-bottom:4px">what you write</div>
        <pre id="ct-src" style="background:#f3f4f6;border:1px solid #d1d5db;border-radius:4px;padding:10px;font-size:11px;line-height:1.5;margin:0">[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user",   "content": "What is 2+2?"},
  {"role": "assistant", "content": "4."}
]</pre>
      </div>
      <div style="flex:1;min-width:260px">
        <div style="font-size:11px;color:#6b7280;margin-bottom:4px">what the model sees (special tokens highlighted)</div>
        <div id="ct-dst" style="background:#1e293b;color:#e2e8f0;border-radius:4px;padding:10px;font-size:11px;line-height:1.8;white-space:pre-wrap;font-family:ui-monospace,monospace"></div>
      </div>
    </div>
    <p class="viz-readout" id="ct-out"></p>
  `;

  const tpl = {
    chatml: {
      render: `<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
4.<|im_end|>`,
      specials: ['<|im_start|>', '<|im_end|>'],
      note: 'ChatML: used by OpenAI (since GPT-4), Qwen, and many others. 2 special tokens mark role boundaries.'
    },
    llama3: {
      render: `<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is 2+2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

4.<|eot_id|>`,
      specials: ['<|begin_of_text|>', '<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>'],
      note: 'Llama-3: 4 special tokens. Header delimiters + end-of-turn. More verbose than ChatML.'
    },
    nanochat: {
      render: `<|bos|><|system_start|>You are a helpful assistant.<|system_end|><|user_start|>What is 2+2?<|user_end|><|assistant_start|>4.<|assistant_end|>`,
      specials: ['<|bos|>', '<|system_start|>', '<|system_end|>', '<|user_start|>', '<|user_end|>', '<|assistant_start|>', '<|assistant_end|>'],
      note: 'nanochat: simple explicit start/end pairs per role. Easy to parse, easy to train from scratch.'
    }
  };

  function render() {
    const key = container.querySelector('#ct-t').value;
    const t = tpl[key];
    let html = t.render.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    t.specials.forEach(s => {
      const esc = s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
      html = html.split(esc).join(`<span style="background:#f59e0b;color:#fff;padding:1px 4px;border-radius:2px">${esc}</span>`);
    });
    container.querySelector('#ct-dst').innerHTML = html;
    const tokenCount = t.render.length; // rough
    container.querySelector('#ct-out').innerHTML = `<b>${t.note}</b> The model never sees the JSON — it sees this flat stream. Loss is computed on every token, but during SFT we mask everything except the assistant's turn.`;
  }
  container.querySelector('#ct-t').addEventListener('change', render);
  render();
});
