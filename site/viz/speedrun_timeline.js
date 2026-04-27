// viz/speedrun_timeline.js
// nanochat speedrun timeline: visualize 3.5h end-to-end.

registerViz('speedrun_timeline', function (container) {
  container.innerHTML = `
    <p class="viz-title">nanochat speedrun: 3.5 hours, end to end</p>
    <p class="viz-sub">Every stage of <code>runs/speedrun.sh</code>, time-scaled. Click any stage for details.</p>
    <svg id="st-svg" viewBox="0 0 800 260" width="800" height="260" style="background:#fff;border:1px solid #e5e7eb;border-radius:4px;max-width:100%"></svg>
    <p class="viz-readout" id="st-out">Click any stage for detail.</p>
  `;
  const stages = [
    { name: 'env setup',     start: 0,    duration: 8,   color: '#a78bfa', detail: 'uv venv, install deps, wandb login. ~8 min.' },
    { name: 'download data', start: 8,    duration: 12,  color: '#60a5fa', detail: 'Download 170 tokenized shards (~17 GB) from S3. Async while tokenizer trains.' },
    { name: 'tok_train',     start: 8,    duration: 10,  color: '#34d399', detail: 'Train BPE tokenizer: 32768 vocab on 2B chars. ~10 min.' },
    { name: 'tok_eval',      start: 18,   duration: 2,   color: '#10b981', detail: 'Compression benchmark vs tiktoken.' },
    { name: 'pretrain (base_train)', start: 20,   duration: 130, color: '#f59e0b', detail: 'The big one. d24 model, ~40B tokens trained, fp8 matmul on H100. ~2h 10min.' },
    { name: 'base_eval',     start: 150,  duration: 15,  color: '#eab308', detail: 'val_bpb + CORE benchmark. ~15 min.' },
    { name: 'SFT',           start: 165,  duration: 20,  color: '#ec4899', detail: 'SmolTalk + identity conversations. Teaches chat format. ~20 min.' },
    { name: 'chat_eval',     start: 185,  duration: 30,  color: '#f43f5e', detail: 'MMLU, ARC, GSM8K, HumanEval. ~30 min.' },
    { name: 'report',        start: 215,  duration: 3,   color: '#6366f1', detail: 'Generate report.md.' },
  ];
  const totalMin = 218;
  const svg = container.querySelector('#st-svg');
  const out = container.querySelector('#st-out');
  function render() {
    const W = 800, padX = 40, H = 40, yOff = 100;
    let html = '';
    // timeline axis
    html += `<line x1="${padX}" y1="${yOff + H + 10}" x2="${W - padX}" y2="${yOff + H + 10}" stroke="#6b7280"/>`;
    for (let t = 0; t <= totalMin; t += 30) {
      const x = padX + (t / totalMin) * (W - 2 * padX);
      const hr = Math.floor(t / 60), mn = t % 60;
      html += `<line x1="${x}" y1="${yOff + H + 8}" x2="${x}" y2="${yOff + H + 14}" stroke="#6b7280"/>`;
      html += `<text x="${x}" y="${yOff + H + 30}" font-size="10" text-anchor="middle" fill="#6b7280">${hr}h${mn}m</text>`;
    }
    // stages (two rows to separate overlapping ones)
    const rowOffsets = {};
    const rowHeight = 32;
    let maxRow = 0;
    stages.forEach((s, idx) => {
      const x = padX + (s.start / totalMin) * (W - 2 * padX);
      const w = (s.duration / totalMin) * (W - 2 * padX);
      // assign a row: check conflicts
      let row = 0;
      while (stages.slice(0, idx).some(other => {
        const myEnd = s.start + s.duration;
        const otherEnd = other.start + other.duration;
        return rowOffsets[other.name] === row && !(s.start >= otherEnd || myEnd <= other.start);
      })) row++;
      rowOffsets[s.name] = row;
      maxRow = Math.max(maxRow, row);
      const y = yOff - row * rowHeight;
      html += `<g data-idx="${idx}" style="cursor:pointer">`;
      html += `<rect x="${x}" y="${y}" width="${w}" height="${H - 6}" fill="${s.color}" rx="4" opacity="0.85"/>`;
      html += `<text x="${x + 4}" y="${y + 22}" font-size="11" fill="#fff" font-weight="bold">${s.name}</text>`;
      html += `</g>`;
    });
    svg.innerHTML = html;
    svg.querySelectorAll('g[data-idx]').forEach(g => {
      g.addEventListener('click', () => {
        const s = stages[+g.dataset.idx];
        out.innerHTML = `<b>${s.name}</b> (starts at ${Math.floor(s.start/60)}h${s.start%60}m, duration ${s.duration} min): ${s.detail}`;
      });
    });
  }
  render();
});
