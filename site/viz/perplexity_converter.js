// viz/perplexity_converter.js
// Live converter between loss / perplexity / bpb / bits-per-word.

registerViz('perplexity_converter', function (container) {
  container.innerHTML = `
    <p class="viz-title">Loss ⟷ perplexity ⟷ bits-per-byte</p>
    <p class="viz-sub">Change any one; the others update. Compare to real model scores below.</p>
    <div class="viz-controls" style="flex-wrap:wrap">
      <label>Cross-entropy loss (nats): <input id="pc-l" type="number" step="0.01" value="2.5" style="width:80px"></label>
      <label>Perplexity: <input id="pc-p" type="number" step="0.1" value="12.18" style="width:80px"></label>
      <label>BPB (bits/byte): <input id="pc-b" type="number" step="0.01" value="0.90" style="width:80px"></label>
      <label>Bytes/token: <input id="pc-bpt" type="number" step="0.1" value="4.0" style="width:60px"></label>
    </div>
    <div id="pc-ruler" style="margin-top:16px"></div>
    <p class="viz-readout" id="pc-out"></p>
  `;
  let updating = false;

  function update(from) {
    if (updating) return;
    updating = true;
    const lEl = container.querySelector('#pc-l');
    const pEl = container.querySelector('#pc-p');
    const bEl = container.querySelector('#pc-b');
    const bptEl = container.querySelector('#pc-bpt');
    const bpt = +bptEl.value || 4;
    let loss;
    if (from === 'l') loss = +lEl.value;
    else if (from === 'p') loss = Math.log(+pEl.value);
    else if (from === 'b') loss = (+bEl.value) * Math.log(2) * bpt;
    else if (from === 'bpt') loss = +lEl.value;
    if (from !== 'l') lEl.value = loss.toFixed(3);
    if (from !== 'p') pEl.value = Math.exp(loss).toFixed(2);
    if (from !== 'b') bEl.value = (loss / Math.log(2) / bpt).toFixed(3);

    // Ruler
    const ruler = container.querySelector('#pc-ruler');
    const anchors = [
      { name: 'random over 50K vocab',  loss: Math.log(50000), y: '#9ca3af' },
      { name: 'bigram model',            loss: 3.0, y: '#9ca3af' },
      { name: 'GPT-2 (124M) on WebText', loss: 3.1, y: '#f59e0b' },
      { name: 'GPT-2 XL',                loss: 2.7, y: '#f59e0b' },
      { name: 'nanochat d24 speedrun',   loss: 2.1, y: '#2563eb' },
      { name: 'GPT-3 (175B)',            loss: 1.9, y: '#10b981' },
      { name: 'GPT-4 (est)',             loss: 1.5, y: '#10b981' },
    ];
    const maxL = 11;
    anchors.sort((a,b)=>a.loss-b.loss);
    let html = '<div style="position:relative;height:40px;background:#f3f4f6;border-radius:4px;margin-top:16px">';
    anchors.forEach(a => {
      const pct = (a.loss / maxL) * 100;
      html += `<div style="position:absolute;left:${pct}%;top:0;bottom:0;width:2px;background:${a.y}" title="${a.name}"></div>`;
      html += `<div style="position:absolute;left:${pct}%;top:42px;font-size:9px;white-space:nowrap;transform:translateX(-50%);color:${a.y}">${a.name}</div>`;
    });
    // You
    const youPct = (loss / maxL) * 100;
    html += `<div style="position:absolute;left:${youPct}%;top:-8px;bottom:-8px;width:4px;background:#dc2626"></div>`;
    html += `<div style="position:absolute;left:${youPct}%;top:-22px;font-size:11px;font-weight:bold;color:#dc2626;transform:translateX(-50%)">you: ${loss.toFixed(2)}</div>`;
    html += '</div><div style="display:flex;justify-content:space-between;font-size:10px;color:#6b7280;margin-top:60px"><span>0 (perfect)</span><span>loss (nats)</span><span>~10 (random guessing)</span></div>';
    ruler.innerHTML = html;

    container.querySelector('#pc-out').innerHTML =
      `<b>loss = ${loss.toFixed(3)} nats = ${(loss/Math.log(2)).toFixed(3)} bits</b>. ` +
      `Perplexity = exp(loss) = <b>${Math.exp(loss).toFixed(2)}</b> = "as uncertain as uniform choice over ${Math.exp(loss).toFixed(0)} options". ` +
      `BPB = <b>${(loss/Math.log(2)/bpt).toFixed(3)}</b> (vocab-invariant).`;
    updating = false;
  }
  container.querySelector('#pc-l').addEventListener('input', () => update('l'));
  container.querySelector('#pc-p').addEventListener('input', () => update('p'));
  container.querySelector('#pc-b').addEventListener('input', () => update('b'));
  container.querySelector('#pc-bpt').addEventListener('input', () => update('bpt'));
  update('l');
});
