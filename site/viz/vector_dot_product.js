// viz/vector_dot_product.js
// Interactive 2D vector dot product.
// User drags two vectors; shows magnitude, angle, dot product sign.
// Builds the "dot product = alignment" intuition critical for attention.

registerViz('vector_dot_product', function (container) {
  container.innerHTML = `
    <p class="viz-title">Dot product = alignment</p>
    <p class="viz-sub">Drag the arrowheads. Dot product is positive when vectors point the same way, zero when perpendicular, negative when opposite. This is the core of attention.</p>
    <svg id="vd-svg" viewBox="-250 -250 500 500" width="500" height="500" style="border:1px solid #e5e7eb;border-radius:4px;background:#fafafa;max-width:100%"></svg>
    <p class="viz-readout" id="vd-out"></p>
  `;
  const svg = container.querySelector('#vd-svg');
  const out = container.querySelector('#vd-out');

  // state: two vectors
  const A = { x: 120, y: -80 };
  const B = { x: 90, y: 100 };

  function render() {
    svg.innerHTML = `
      <defs>
        <marker id="a-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#ef4444"/>
        </marker>
        <marker id="b-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#2563eb"/>
        </marker>
      </defs>
      <line x1="-250" y1="0" x2="250" y2="0" stroke="#d1d5db"/>
      <line x1="0" y1="-250" x2="0" y2="250" stroke="#d1d5db"/>
      <line x1="0" y1="0" x2="${A.x}" y2="${-A.y}" stroke="#ef4444" stroke-width="3" marker-end="url(#a-arrow)"/>
      <line x1="0" y1="0" x2="${B.x}" y2="${-B.y}" stroke="#2563eb" stroke-width="3" marker-end="url(#b-arrow)"/>
      <circle cx="${A.x}" cy="${-A.y}" r="8" fill="#ef4444" data-which="A" style="cursor:grab"/>
      <circle cx="${B.x}" cy="${-B.y}" r="8" fill="#2563eb" data-which="B" style="cursor:grab"/>
      <text x="${A.x + 12}" y="${-A.y - 12}" fill="#ef4444" font-size="14" font-weight="bold">A</text>
      <text x="${B.x + 12}" y="${-B.y - 12}" fill="#2563eb" font-size="14" font-weight="bold">B</text>
    `;
    const dot = A.x * B.x + A.y * B.y;
    const magA = Math.hypot(A.x, A.y);
    const magB = Math.hypot(B.x, B.y);
    const cosTheta = dot / (magA * magB);
    const theta = Math.acos(Math.max(-1, Math.min(1, cosTheta))) * 180 / Math.PI;
    const color = dot > 0 ? '#15803d' : dot < 0 ? '#b91c1c' : '#6b7280';
    out.innerHTML =
      `A = (${A.x.toFixed(0)}, ${A.y.toFixed(0)})   |A| = ${magA.toFixed(1)}\n` +
      `B = (${B.x.toFixed(0)}, ${B.y.toFixed(0)})   |B| = ${magB.toFixed(1)}\n` +
      `angle between = ${theta.toFixed(1)}°\n` +
      `<span style="color:${color};font-weight:bold">A · B = ${dot.toFixed(0)}</span>   ` +
      (dot > 0 ? '(aligned)' : dot < 0 ? '(opposite)' : '(perpendicular)');
    hookDrag();
  }

  function hookDrag() {
    const circles = svg.querySelectorAll('circle');
    circles.forEach(c => {
      c.addEventListener('mousedown', e => {
        e.preventDefault();
        const which = c.dataset.which;
        const rect = svg.getBoundingClientRect();
        const toCoord = (ev) => {
          const x = (ev.clientX - rect.left) / rect.width * 500 - 250;
          const y = -((ev.clientY - rect.top) / rect.height * 500 - 250);
          return { x, y };
        };
        const onMove = ev => {
          const p = toCoord(ev);
          if (which === 'A') { A.x = p.x; A.y = p.y; } else { B.x = p.x; B.y = p.y; }
          render();
        };
        const onUp = () => {
          window.removeEventListener('mousemove', onMove);
          window.removeEventListener('mouseup', onUp);
        };
        window.addEventListener('mousemove', onMove);
        window.addEventListener('mouseup', onUp);
      });
    });
  }
  render();
});
