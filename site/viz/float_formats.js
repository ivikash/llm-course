// viz/float_formats.js
// Visualize fp32 vs bf16 vs fp16 vs fp8 bit layouts and ranges.

registerViz('float_formats', function (container) {
  container.innerHTML = `
    <p class="viz-title">Float formats compared</p>
    <p class="viz-sub">Each bar shows the bits: sign | exponent | mantissa. And the range/precision trade-off.</p>
    <div id="ff-bars"></div>
    <p class="viz-readout" id="ff-out" style="margin-top:12px">Hover a format for details.</p>
  `;
  const formats = [
    { name: 'fp32',  sign: 1, exp: 8, mant: 23, range: '±3.4×10³⁸', prec: '~7 dec', use: 'default scientific computing, master weights in mixed precision' },
    { name: 'bf16',  sign: 1, exp: 8, mant: 7,  range: '±3.4×10³⁸', prec: '~2 dec', use: 'modern LLM training default; fp32 range + half the memory' },
    { name: 'fp16',  sign: 1, exp: 5, mant: 10, range: '±6.5×10⁴',  prec: '~3 dec', use: 'older mixed precision; narrow range → gradient underflow' },
    { name: 'fp8 e4m3', sign: 1, exp: 4, mant: 3, range: '±448',      prec: '~0.4 dec', use: 'H100+ forward activations; needs dynamic scaling' },
    { name: 'fp8 e5m2', sign: 1, exp: 5, mant: 2, range: '±57344',   prec: '~0.2 dec', use: 'H100+ gradients; more range, less precision' },
    { name: 'int8',  sign: 1, exp: 0, mant: 7,  range: '-128..127', prec: 'integer', use: 'post-training weight quantization for inference' },
    { name: 'int4',  sign: 1, exp: 0, mant: 3,  range: '-8..7',     prec: 'integer', use: 'aggressive quantization; GGUF/GPTQ/AWQ models' },
  ];
  const maxBits = 32;
  const bars = formats.map(f => {
    const total = f.sign + f.exp + f.mant;
    const scale = 560 / maxBits;
    const signW = f.sign * scale;
    const expW = f.exp * scale;
    const mantW = f.mant * scale;
    return `<div data-name="${f.name}" style="margin:8px 0;cursor:pointer">
      <div style="font-family:ui-monospace,monospace;font-size:13px;margin-bottom:2px">${f.name} (${total} bits)</div>
      <div style="display:flex;align-items:center;height:24px;border:1px solid #d1d5db;border-radius:3px;overflow:hidden">
        <div style="width:${signW}px;background:#fca5a5;height:100%;text-align:center;color:#7f1d1d;font-size:10px;line-height:24px">S</div>
        ${f.exp > 0 ? `<div style="width:${expW}px;background:#93c5fd;height:100%;text-align:center;color:#1e3a8a;font-size:10px;line-height:24px">exponent (${f.exp})</div>` : ''}
        <div style="width:${mantW}px;background:#86efac;height:100%;text-align:center;color:#14532d;font-size:10px;line-height:24px">mantissa (${f.mant})</div>
      </div>
    </div>`;
  }).join('');
  container.querySelector('#ff-bars').innerHTML = bars;
  const out = container.querySelector('#ff-out');
  container.querySelectorAll('[data-name]').forEach(el => {
    el.addEventListener('mouseenter', () => {
      const f = formats.find(x => x.name === el.dataset.name);
      out.innerHTML = `<b>${f.name}</b>: range ${f.range}, precision ${f.prec}. Use: ${f.use}`;
    });
  });
});
