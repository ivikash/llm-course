// viz/rag_pipeline.js
// RAG pipeline visualization: user query → embed → retrieve → augment → answer.

registerViz('rag_pipeline', function (container) {
  container.innerHTML = `
    <p class="viz-title">RAG pipeline</p>
    <p class="viz-sub">Type a query. Watch it embed, retrieve top-k chunks, assemble the prompt, and generate an answer.</p>
    <div class="viz-controls">
      <label>Query: <input id="rp-q" type="text" value="What is the refund policy for Pro subscribers?" style="width:400px"></label>
      <label>top-k: <input id="rp-k" type="range" min="1" max="5" value="3"><span id="rp-k-out">3</span></label>
      <button id="rp-run">Run</button>
    </div>
    <div id="rp-pipeline" style="display:flex;flex-direction:column;gap:8px;margin-top:16px"></div>
  `;
  // Toy knowledge base
  const docs = [
    { id: 1, text: 'Pro subscribers can cancel at any time. Cancellation takes effect at end of billing period.' },
    { id: 2, text: 'Pro subscribers can request a refund within 30 days of purchase. Refunds are processed in 5-7 business days.' },
    { id: 3, text: 'Free users have limited daily queries. Upgrading to Pro removes this limit.' },
    { id: 4, text: 'Enterprise plans include SSO, audit logs, and dedicated support.' },
    { id: 5, text: 'All new users get a 14-day free trial of Pro with no credit card required.' },
    { id: 6, text: 'Billing occurs monthly. Failed payments trigger a 7-day grace period before downgrade.' },
    { id: 7, text: 'Our documentation is available in English, Spanish, French, German, and Japanese.' },
    { id: 8, text: 'API rate limits: 60 requests/min for Pro, 600/min for Enterprise.' },
  ];
  // Toy embedding: bag of words
  function embed(text) {
    const words = text.toLowerCase().match(/\w+/g) || [];
    const v = {};
    for (const w of words) v[w] = (v[w] || 0) + 1;
    return v;
  }
  function cosine(a, b) {
    const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
    let dot = 0, na = 0, nb = 0;
    for (const k of keys) { const va = a[k] || 0, vb = b[k] || 0; dot += va * vb; na += va*va; nb += vb*vb; }
    return dot / (Math.sqrt(na * nb) || 1);
  }
  async function run() {
    const query = container.querySelector('#rp-q').value;
    const k = +container.querySelector('#rp-k').value;
    container.querySelector('#rp-k-out').textContent = k;
    const pipe = container.querySelector('#rp-pipeline');
    pipe.innerHTML = '';

    async function addStep(title, body, color) {
      const box = document.createElement('div');
      box.style.cssText = `border-left:4px solid ${color};padding:8px 12px;background:#fafafa;border-radius:4px;opacity:0;transition:opacity 0.4s`;
      box.innerHTML = `<div style="font-size:12px;color:${color};font-weight:bold;margin-bottom:4px">${title}</div><div style="font-size:13px">${body}</div>`;
      pipe.appendChild(box);
      await new Promise(r => setTimeout(r, 100));
      box.style.opacity = 1;
      await new Promise(r => setTimeout(r, 500));
    }
    await addStep('1. Query', `"${query}"`, '#6b7280');
    await addStep('2. Embed query', 'Convert query to a vector (~1536 floats with OpenAI embedding-3-small). Here we use bag-of-words.', '#2563eb');
    const qVec = embed(query);
    await addStep('3. Search vector DB', `Compute similarity against all ${docs.length} chunks...`, '#8b5cf6');
    const scored = docs.map(d => ({ ...d, sim: cosine(qVec, embed(d.text)) }));
    scored.sort((a, b) => b.sim - a.sim);
    const top = scored.slice(0, k);
    const listHtml = top.map(d => `<div style="padding:4px 8px;background:#fff;border:1px solid #e5e7eb;border-radius:4px;margin-top:2px;font-size:12px"><b>chunk ${d.id}</b> (sim=${d.sim.toFixed(2)}): ${d.text}</div>`).join('');
    await addStep(`4. Top-${k} chunks retrieved`, listHtml, '#059669');
    const context = top.map(d => `[${d.id}] ${d.text}`).join('\n');
    await addStep('5. Assemble prompt', `<pre style="font-size:11px;background:#fff;padding:8px;border-radius:4px;white-space:pre-wrap">Answer using only this context:\n${context}\n\nQuestion: ${query}\nAnswer:</pre>`, '#d97706');
    // simulated answer
    let answer = 'Based on the context...';
    if (top[0].text.toLowerCase().includes('refund')) {
      answer = 'Pro subscribers can request a refund within 30 days of purchase, processed in 5-7 business days. (Source: chunk ' + top[0].id + ')';
    } else if (top[0].text.toLowerCase().includes('rate')) {
      answer = 'Pro users get 60 requests/min; Enterprise gets 600. (Source: chunk ' + top[0].id + ')';
    } else {
      answer = 'Based on chunk ' + top[0].id + ': ' + top[0].text;
    }
    await addStep('6. LLM generates answer', answer, '#dc2626');
  }
  container.querySelector('#rp-run').addEventListener('click', run);
  run();
});
