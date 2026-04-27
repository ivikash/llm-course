// viz/agent_eval_scorecard.js
// Live agent benchmark scorecard across common benchmarks.

registerViz('agent_eval_scorecard', function (container) {
  container.innerHTML = `
    <p class="viz-title">Agent benchmark scorecard</p>
    <p class="viz-sub">Hover any benchmark for the task. Click a model column to highlight.</p>
    <div class="viz-controls">
      <label>Year: <input id="ae-y" type="range" min="2020" max="2026" value="2026"><span id="ae-y-out">2026</span></label>
    </div>
    <div id="ae-table" style="overflow-x:auto"></div>
    <p class="viz-readout" id="ae-out">Hover a benchmark or model.</p>
  `;

  const models = [
    { name: 'GPT-3.5',    year: 2022 },
    { name: 'GPT-4',      year: 2023 },
    { name: 'Claude 3.5', year: 2024 },
    { name: 'GPT-4o',     year: 2024 },
    { name: 'o1',         year: 2024 },
    { name: 'DeepSeek-R1',year: 2025 },
    { name: 'o3',         year: 2025 },
    { name: 'Claude 4',    year: 2026 },
  ];

  const benchmarks = [
    { name: 'MMLU',          desc: 'General knowledge across 57 subjects. 4-choice MCQ.',
      scores: { 'GPT-3.5': 70, 'GPT-4': 86, 'Claude 3.5': 88, 'GPT-4o': 88, 'o1': 91, 'DeepSeek-R1': 89, 'o3': 93, 'Claude 4': 95 } },
    { name: 'GPQA',           desc: 'PhD-level science. Multiple choice, very hard.',
      scores: { 'GPT-3.5': 28, 'GPT-4': 36, 'Claude 3.5': 59, 'GPT-4o': 54, 'o1': 78, 'DeepSeek-R1': 72, 'o3': 88, 'Claude 4': 92 } },
    { name: 'HumanEval',      desc: 'Python coding. 164 problems. pass@1.',
      scores: { 'GPT-3.5': 48, 'GPT-4': 67, 'Claude 3.5': 93, 'GPT-4o': 91, 'o1': 89, 'DeepSeek-R1': 90, 'o3': 97, 'Claude 4': 98 } },
    { name: 'SWE-Bench Verif.', desc: 'Real GitHub issues. Agent must ship a passing PR.',
      scores: { 'GPT-3.5': 0.2, 'GPT-4': 1.7, 'Claude 3.5': 49, 'GPT-4o': 33, 'o1': 41, 'DeepSeek-R1': 51, 'o3': 71, 'Claude 4': 76 } },
    { name: 'GSM8K',           desc: 'Grade-school math word problems.',
      scores: { 'GPT-3.5': 57, 'GPT-4': 92, 'Claude 3.5': 95, 'GPT-4o': 94, 'o1': 95, 'DeepSeek-R1': 95, 'o3': 97, 'Claude 4': 98 } },
    { name: 'AIME',             desc: 'High-school math olympiad. Very hard.',
      scores: { 'GPT-3.5': 2, 'GPT-4': 13, 'Claude 3.5': 16, 'GPT-4o': 13, 'o1': 83, 'DeepSeek-R1': 79, 'o3': 92, 'Claude 4': 88 } },
    { name: 'GAIA',              desc: 'Real-world assistant tasks requiring tools.',
      scores: { 'GPT-3.5': 10, 'GPT-4': 15, 'Claude 3.5': 32, 'GPT-4o': 35, 'o1': 48, 'DeepSeek-R1': 42, 'o3': 58, 'Claude 4': 62 } },
    { name: 'ARC-AGI',            desc: 'Fluid reasoning puzzles. Humans ~85%.',
      scores: { 'GPT-3.5': 0, 'GPT-4': 2, 'Claude 3.5': 13, 'GPT-4o': 5, 'o1': 25, 'DeepSeek-R1': 15, 'o3': 87, 'Claude 4': 60 } },
  ];

  let highlightModel = null;

  function render() {
    const yr = +container.querySelector('#ae-y').value;
    container.querySelector('#ae-y-out').textContent = yr;
    const shownModels = models.filter(m => m.year <= yr);
    let html = `<table style="border-collapse:collapse;font-size:12px;min-width:100%">
      <thead><tr><th style="padding:8px;text-align:left;background:#f3f4f6">Benchmark</th>`;
    shownModels.forEach(m => {
      const sel = m.name === highlightModel;
      html += `<th data-model="${m.name}" style="padding:8px;background:${sel?'#dbeafe':'#f3f4f6'};cursor:pointer;font-size:11px;min-width:70px">${m.name}<br><span style="color:#9ca3af;font-weight:normal">${m.year}</span></th>`;
    });
    html += `</tr></thead><tbody>`;
    benchmarks.forEach(b => {
      html += `<tr data-bench="${b.name}" style="cursor:pointer"><td style="padding:6px 8px;border-top:1px solid #e5e7eb;font-weight:bold">${b.name}</td>`;
      shownModels.forEach(m => {
        const s = b.scores[m.name];
        const hue = s > 80 ? 120 : s > 50 ? 60 : 0;
        const bg = `hsl(${hue}, 70%, ${90 - Math.min(60, s / 2)}%)`;
        const fg = s > 60 ? '#14532d' : s > 30 ? '#78350f' : '#7f1d1d';
        html += `<td style="padding:6px 8px;border-top:1px solid #e5e7eb;background:${bg};color:${fg};text-align:center;font-family:ui-monospace,monospace">${s}</td>`;
      });
      html += `</tr>`;
    });
    html += `</tbody></table>`;
    container.querySelector('#ae-table').innerHTML = html;
    container.querySelectorAll('tr[data-bench]').forEach(row => {
      row.addEventListener('mouseenter', () => {
        const b = benchmarks.find(b => b.name === row.dataset.bench);
        container.querySelector('#ae-out').innerHTML = `<b>${b.name}</b>: ${b.desc}`;
      });
    });
    container.querySelectorAll('th[data-model]').forEach(th => {
      th.addEventListener('click', () => {
        highlightModel = highlightModel === th.dataset.model ? null : th.dataset.model;
        render();
      });
    });
  }
  container.querySelector('#ae-y').addEventListener('input', render);
  render();
});
