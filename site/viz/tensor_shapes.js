// viz/tensor_shapes.js
// Interactive 3D tensor visualizer using Three.js.
// User picks dims (B, T, C); sees the shape rendered as a box of cubes.

registerViz('tensor_shapes', async function (container) {
  container.innerHTML = `
    <p class="viz-title">Tensor shapes in 3D</p>
    <p class="viz-sub">A tensor of shape (B, T, C) visualized. Classic LLM input: batch × sequence × channels. Slide the dims.</p>
    <div class="viz-controls">
      <label>B (batch): <input id="ts-b" type="range" min="1" max="16" value="4"><span id="ts-b-out">4</span></label>
      <label>T (seq): <input id="ts-t" type="range" min="1" max="32" value="8"><span id="ts-t-out">8</span></label>
      <label>C (channels): <input id="ts-c" type="range" min="1" max="32" value="12"><span id="ts-c-out">12</span></label>
    </div>
    <div id="ts-canvas" style="width:100%;height:360px;background:#f9fafb;border-radius:4px;overflow:hidden"></div>
    <p class="viz-readout" id="ts-out"></p>
  `;

  await ensureThree();
  const THREE = window.THREE;

  const canvas = container.querySelector('#ts-canvas');
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h);
  canvas.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf9fafb);
  const camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 1000);
  camera.position.set(28, 22, 32);
  camera.lookAt(0, 0, 0);

  const ambient = new THREE.AmbientLight(0xffffff, 0.5);
  const dir = new THREE.DirectionalLight(0xffffff, 0.7);
  dir.position.set(10, 20, 15);
  scene.add(ambient, dir);

  let group = new THREE.Group();
  scene.add(group);

  const bEl = container.querySelector('#ts-b');
  const tEl = container.querySelector('#ts-t');
  const cEl = container.querySelector('#ts-c');
  const bOut = container.querySelector('#ts-b-out');
  const tOut = container.querySelector('#ts-t-out');
  const cOut = container.querySelector('#ts-c-out');
  const out = container.querySelector('#ts-out');

  function build() {
    scene.remove(group);
    group = new THREE.Group();
    const B = +bEl.value, T = +tEl.value, C = +cEl.value;
    bOut.textContent = B; tOut.textContent = T; cOut.textContent = C;

    const geom = new THREE.BoxGeometry(0.9, 0.9, 0.9);
    const offX = -T / 2, offY = -C / 2, offZ = -B / 2;
    // color by batch index for visual separation
    for (let b = 0; b < B; b++) {
      const hue = (b / B) * 0.7 + 0.05;
      const color = new THREE.Color().setHSL(hue, 0.6, 0.55);
      const mat = new THREE.MeshPhongMaterial({ color, transparent: true, opacity: 0.85 });
      for (let t = 0; t < T; t++) {
        for (let c = 0; c < C; c++) {
          const mesh = new THREE.Mesh(geom, mat);
          mesh.position.set(offX + t, offY + c, offZ + b);
          group.add(mesh);
        }
      }
    }
    scene.add(group);
    const total = B * T * C;
    out.textContent = `shape = (${B}, ${T}, ${C}) — ${total.toLocaleString()} scalars. ` +
      `Memory at fp32: ${(total * 4).toLocaleString()} bytes.`;
  }

  // orbit: gentle rotation
  let t = 0;
  function loop() {
    requestAnimationFrame(loop);
    t += 0.005;
    group.rotation.y = Math.sin(t) * 0.5;
    renderer.render(scene, camera);
  }
  loop();

  [bEl, tEl, cEl].forEach(el => el.addEventListener('input', build));
  build();

  // handle resize
  new ResizeObserver(() => {
    const nw = canvas.clientWidth, nh = canvas.clientHeight;
    if (nw && nh) { renderer.setSize(nw, nh); camera.aspect = nw / nh; camera.updateProjectionMatrix(); }
  }).observe(canvas);
});
