// viz/sft_loss_mask.js
// Shows how SFT masks the loss on user tokens vs assistant tokens.

registerViz('sft_loss_mask', function (container) {
  container.innerHTML = `
    <p class="viz-title">SFT loss masking</p>
    <p class="viz-sub">A chat conversation, tokenized. Green = 'assistant' tokens (loss IS computed). Red = 'user' tokens (loss is IGNORED). Without masking, the model would try to predict user inputs too — chaos.</p>
    <div class="viz-controls">
      <label>Scenario:
        <select id="sm-s">
          <option value="short">Short Q&A</option>
          <option value="long" selected>Multi-turn chat</option>
          <option value="tool">With tool use</option>
        </select>
      </label>
    </div>
    <div id="sm-tokens" style="padding:16px;background:#fafafa;border-radius:6px;line-height:2.2;min-height:100px"></div>
    <p class="viz-readout" id="sm-out"></p>
  `;
  const scenarios = {
    short: [
      ['<|bos|>', 'S'], ['<|user_start|>', 'S'],
      ['What', 'u'], ['is', 'u'], ['2+2', 'u'], ['?', 'u'],
      ['<|user_end|>', 'S'], ['<|assistant_start|>', 'S'],
      ['4', 'a'], ['.', 'a'],
      ['<|assistant_end|>', 'S'],
    ],
    long: [
      ['<|bos|>', 'S'], ['<|user_start|>', 'S'],
      ['Hi', 'u'], ['there', 'u'], ['!', 'u'],
      ['<|user_end|>', 'S'], ['<|assistant_start|>', 'S'],
      ['Hello', 'a'], ['!', 'a'], ['How', 'a'], ['can', 'a'], ['I', 'a'], ['help', 'a'], ['?', 'a'],
      ['<|assistant_end|>', 'S'], ['<|user_start|>', 'S'],
      ['Explain', 'u'], ['gravity', 'u'], ['.', 'u'],
      ['<|user_end|>', 'S'], ['<|assistant_start|>', 'S'],
      ['Gravity', 'a'], ['is', 'a'], ['the', 'a'], ['force', 'a'], ['pulling', 'a'], ['masses', 'a'], ['together', 'a'], ['.', 'a'],
      ['<|assistant_end|>', 'S'],
    ],
    tool: [
      ['<|bos|>', 'S'], ['<|user_start|>', 'S'],
      ['Compute', 'u'], ['237*491', 'u'],
      ['<|user_end|>', 'S'], ['<|assistant_start|>', 'S'],
      ['<|python_start|>', 'a'], ['237*491', 'a'], ['<|python_end|>', 'a'],
      ['<|output_start|>', 'S'], ['116367', 'S'], ['<|output_end|>', 'S'],
      ['The', 'a'], ['answer', 'a'], ['is', 'a'], ['116367', 'a'], ['.', 'a'],
      ['<|assistant_end|>', 'S'],
    ],
  };
  function render() {
    const s = container.querySelector('#sm-s').value;
    const toks = scenarios[s];
    const html = toks.map(([t, type], i) => {
      const color = type === 'a' ? '#bbf7d0' : type === 'u' ? '#fecaca' : '#e5e7eb';
      const textColor = type === 'a' ? '#14532d' : type === 'u' ? '#7f1d1d' : '#374151';
      const mask = type === 'a' ? '1' : '0';
      return `<span title="loss_mask = ${mask}" style="background:${color};color:${textColor};padding:3px 8px;border-radius:4px;margin:2px;display:inline-block;font-family:ui-monospace,monospace;font-size:12px">${t}<span style="font-size:9px;opacity:0.6;margin-left:4px">${mask}</span></span>`;
    }).join('');
    container.querySelector('#sm-tokens').innerHTML = html;
    const a = toks.filter(t => t[1] === 'a').length;
    const u = toks.filter(t => t[1] === 'u').length;
    const total = toks.length;
    container.querySelector('#sm-out').textContent =
      `Total tokens: ${total}. Assistant tokens (loss computed): ${a}. User tokens (ignored): ${u}. Special tokens: ${total - a - u}. ` +
      `Only ${a}/${total} (${((a/total)*100).toFixed(0)}%) of tokens contribute to SFT loss.`;
  }
  container.querySelector('#sm-s').addEventListener('change', render);
  render();
});
