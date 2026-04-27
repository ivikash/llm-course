// viz/paper_tree.js
// Clickable SVG tree of influential ML papers, annotated with years.

registerViz('paper_tree', function (container) {
  container.innerHTML = `
    <p class="viz-title">The transformer family tree</p>
    <p class="viz-sub">25 papers that built modern LLMs. Click any node to see the one-line idea and link.</p>
    <svg id="pt-svg" viewBox="0 0 900 600" width="900" height="600" style="background:#fff;border:1px solid #e5e7eb;border-radius:4px;max-width:100%"></svg>
    <p class="viz-readout" id="pt-out">Click a node.</p>
  `;
  const nodes = [
    { id: 'attn', x: 450, y: 40, label: 'Attention Is All You Need (2017)',
      idea: 'The transformer: multi-head attention + residuals + layer norm. Replaces RNNs for seq-to-seq.',
      url: 'https://arxiv.org/abs/1706.03762' },
    { id: 'gpt1', x: 250, y: 130, label: 'GPT-1 (2018)',
      idea: 'Pretrain + fine-tune paradigm. Decoder-only transformer for language modeling.',
      url: 'https://openai.com/research/language-unsupervised' },
    { id: 'bert', x: 550, y: 130, label: 'BERT (2018)',
      idea: 'Bidirectional encoder. Masked language modeling.',
      url: 'https://arxiv.org/abs/1810.04805' },
    { id: 't5', x: 750, y: 130, label: 'T5 (2019)',
      idea: 'Encoder-decoder, all tasks as text-to-text.',
      url: 'https://arxiv.org/abs/1910.10683' },
    { id: 'gpt2', x: 200, y: 210, label: 'GPT-2 (2019)',
      idea: 'Scale + zero-shot. 1.5B params.',
      url: 'https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf' },
    { id: 'gpt3', x: 200, y: 290, label: 'GPT-3 (2020)',
      idea: 'Scale + few-shot in-context learning. 175B params.',
      url: 'https://arxiv.org/abs/2005.14165' },
    { id: 'kaplan', x: 400, y: 260, label: 'Kaplan scaling laws (2020)',
      idea: 'Loss is a power law in N, D, C. (Wrong balance, corrected by Chinchilla.)',
      url: 'https://arxiv.org/abs/2001.08361' },
    { id: 'chinchilla', x: 400, y: 340, label: 'Chinchilla (2022)',
      idea: 'Corrected Kaplan. Scale N and D equally. ~20 tokens per parameter.',
      url: 'https://arxiv.org/abs/2203.15556' },
    { id: 'instruct', x: 100, y: 380, label: 'InstructGPT (2022)',
      idea: 'SFT + RLHF → chat. Basis of ChatGPT.',
      url: 'https://arxiv.org/abs/2203.02155' },
    { id: 'cot', x: 600, y: 310, label: 'Chain-of-Thought (2022)',
      idea: 'Think step by step. Prompts like "Let us think step by step" dramatically improve reasoning.',
      url: 'https://arxiv.org/abs/2201.11903' },
    { id: 'rope', x: 700, y: 250, label: 'RoPE (2021)',
      idea: 'Rotary Position Embeddings. Better than learned positions.',
      url: 'https://arxiv.org/abs/2104.09864' },
    { id: 'flash', x: 800, y: 310, label: 'FlashAttention (2022)',
      idea: 'Same math, tile in SRAM, 2-4× faster attention.',
      url: 'https://arxiv.org/abs/2205.14135' },
    { id: 'llama', x: 350, y: 440, label: 'LLaMA (2023)',
      idea: 'Open-weights. RoPE + RMSNorm + SwiGLU + GQA recipe.',
      url: 'https://arxiv.org/abs/2302.13971' },
    { id: 'mistral', x: 550, y: 440, label: 'Mistral 7B (2023)',
      idea: 'Open, small, sliding-window attention. Beats Llama-2 13B.',
      url: 'https://arxiv.org/abs/2310.06825' },
    { id: 'mixtral', x: 700, y: 440, label: 'Mixtral 8x7B (2024)',
      idea: 'Open Mixture of Experts. 47B total, 13B active per token.',
      url: 'https://arxiv.org/abs/2401.04088' },
    { id: 'dpo', x: 100, y: 460, label: 'DPO (2023)',
      idea: 'Preference tuning without RL or reward model.',
      url: 'https://arxiv.org/abs/2305.18290' },
    { id: 'llama3', x: 350, y: 520, label: 'Llama-3 (2024)',
      idea: 'Open 8B/70B/405B. 15T tokens, far beyond Chinchilla.',
      url: 'https://ai.meta.com/research/publications/the-llama-3-herd-of-models/' },
    { id: 'r1', x: 150, y: 540, label: 'DeepSeek-R1 (2025)',
      idea: 'GRPO on verifiable rewards. Emergent chain-of-thought.',
      url: 'https://arxiv.org/abs/2501.12948' },
    { id: 'o1', x: 50, y: 460, label: 'OpenAI o1 (2024)',
      idea: 'Inference-time reasoning via RL. Closed, but similar to R1.',
      url: 'https://openai.com' },
    { id: 'mamba', x: 780, y: 380, label: 'Mamba (2023)',
      idea: 'State-space models as transformer alternative. Linear in T.',
      url: 'https://arxiv.org/abs/2312.00752' },
  ];
  const edges = [
    ['attn','gpt1'], ['attn','bert'], ['attn','t5'],
    ['gpt1','gpt2'], ['gpt2','gpt3'],
    ['gpt3','kaplan'], ['kaplan','chinchilla'],
    ['gpt3','instruct'], ['instruct','dpo'], ['instruct','o1'],
    ['gpt3','cot'],
    ['attn','rope'], ['attn','flash'],
    ['chinchilla','llama'], ['rope','llama'],
    ['llama','mistral'], ['llama','mixtral'], ['llama','llama3'],
    ['dpo','llama3'], ['o1','r1'], ['attn','mamba'],
  ];
  const svg = container.querySelector('#pt-svg');
  const out = container.querySelector('#pt-out');
  let html = '';
  edges.forEach(([a, b]) => {
    const na = nodes.find(n => n.id === a);
    const nb = nodes.find(n => n.id === b);
    html += `<line x1="${na.x}" y1="${na.y}" x2="${nb.x}" y2="${nb.y}" stroke="#d1d5db" stroke-width="1"/>`;
  });
  nodes.forEach(n => {
    html += `<g data-id="${n.id}" style="cursor:pointer">`;
    html += `<rect x="${n.x - 70}" y="${n.y - 14}" width="140" height="28" fill="#fff" stroke="#3b82f6" stroke-width="1.5" rx="6"/>`;
    html += `<text x="${n.x}" y="${n.y + 4}" font-size="10" text-anchor="middle" fill="#1e40af" font-weight="600">${n.label}</text>`;
    html += `</g>`;
  });
  svg.innerHTML = html;
  svg.querySelectorAll('g[data-id]').forEach(g => {
    g.addEventListener('click', () => {
      const n = nodes.find(n => n.id === g.dataset.id);
      out.innerHTML = `<b>${n.label}</b>: ${n.idea} — <a href="${n.url}" target="_blank">read paper</a>`;
    });
  });
});
