// viz/bigram_generator.js
// Live character-level bigram model. User picks a seed character;
// model generates text from a pre-built transition table (embedded).
// Shows: the actual probabilistic generation mechanism.

registerViz('bigram_generator', function (container) {
  container.innerHTML = `
    <p class="viz-title">Live bigram generator</p>
    <p class="viz-sub">A character-level Markov model trained on Shakespeare. Same idea as the Python in this lesson, running in your browser.</p>
    <div class="viz-controls">
      <label>Seed: <input id="bg-seed" type="text" value="R" maxlength="1" style="width:32px"></label>
      <label>Length: <input id="bg-len" type="range" min="50" max="2000" value="400"><span id="bg-len-out">400</span></label>
      <label>Temperature: <input id="bg-temp" type="range" min="0.1" max="3" step="0.1" value="1"><span id="bg-temp-out">1</span></label>
      <button id="bg-gen">Generate</button>
    </div>
    <pre class="viz-readout" id="bg-out" style="max-height:240px;overflow:auto;white-space:pre-wrap"></pre>
    <p class="viz-sub" style="margin-top:8px">Try temperature 0.3 (repetitive), 1.0 (Shakespeare-like), 2.0 (chaotic).</p>
  `;

  // Tiny embedded Shakespeare sample to build transitions (~1KB).
  // This is shipped in-browser; no training needed.
  const sample = `ROMEO: But soft, what light through yonder window breaks? It is the east, and Juliet is the sun. Arise, fair sun, and kill the envious moon.
JULIET: O Romeo, Romeo! wherefore art thou Romeo? Deny thy father and refuse thy name.
MERCUTIO: A plague o' both your houses! They have made worms' meat of me.
HAMLET: To be, or not to be, that is the question: whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune.
MACBETH: Is this a dagger which I see before me, the handle toward my hand? Come, let me clutch thee.
IAGO: I am not what I am. Demand me nothing. What you know, you know. From this time forth I never will speak word.
KING LEAR: Blow, winds, and crack your cheeks! Rage! Blow! You cataracts and hurricanoes, spout till you have drenched our steeples.
PORTIA: The quality of mercy is not strained. It droppeth as the gentle rain from heaven upon the place beneath.
PROSPERO: We are such stuff as dreams are made on, and our little life is rounded with a sleep.
BENEDICK: When I said I would die a bachelor, I did not think I should live till I were married.
PUCK: Lord, what fools these mortals be! I'll put a girdle round about the earth in forty minutes.
OTHELLO: Speak of me as I am. Nothing extenuate, nor set down aught in malice. Then must you speak of one that loved not wisely but too well.`;

  // Build bigram counts
  const counts = {};
  for (let i = 0; i < sample.length - 1; i++) {
    const a = sample[i], b = sample[i + 1];
    counts[a] = counts[a] || {};
    counts[a][b] = (counts[a][b] || 0) + 1;
  }

  function sampleNext(ch, temperature) {
    const next = counts[ch];
    if (!next) return sample[Math.floor(Math.random() * sample.length)];
    const chars = Object.keys(next);
    const total = chars.reduce((s, c) => s + next[c], 0);
    // apply temperature by raising probs to 1/T then renormalizing
    const weights = chars.map(c => Math.pow(next[c] / total, 1 / Math.max(0.01, temperature)));
    const wSum = weights.reduce((a, b) => a + b, 0);
    let r = Math.random() * wSum;
    for (let i = 0; i < chars.length; i++) {
      r -= weights[i];
      if (r <= 0) return chars[i];
    }
    return chars[chars.length - 1];
  }

  const out = container.querySelector('#bg-out');
  const seedEl = container.querySelector('#bg-seed');
  const lenEl = container.querySelector('#bg-len');
  const tempEl = container.querySelector('#bg-temp');
  const lenOut = container.querySelector('#bg-len-out');
  const tempOut = container.querySelector('#bg-temp-out');
  const btn = container.querySelector('#bg-gen');

  function generate() {
    const seed = (seedEl.value || 'R')[0];
    const n = +lenEl.value;
    const T = +tempEl.value;
    let cur = seed;
    let result = cur;
    for (let i = 0; i < n; i++) {
      const next = sampleNext(cur, T);
      result += next;
      cur = next;
    }
    out.textContent = result;
  }

  lenEl.addEventListener('input', () => { lenOut.textContent = lenEl.value; });
  tempEl.addEventListener('input', () => { tempOut.textContent = tempEl.value; });
  btn.addEventListener('click', generate);
  generate();
});
