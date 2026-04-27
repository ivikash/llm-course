// site/viz/loader.js
// Finds <div data-viz="name" ...> placeholders in rendered markdown
// and mounts the corresponding interactive visualization.

window.VIZ_REGISTRY = window.VIZ_REGISTRY || {};

window.registerViz = function (name, fn) {
  window.VIZ_REGISTRY[name] = fn;
};

window.mountAllViz = async function (rootEl) {
  const placeholders = rootEl.querySelectorAll('[data-viz]');
  for (const el of placeholders) {
    const name = el.getAttribute('data-viz');
    // parse props from data-* attributes
    const props = {};
    for (const attr of el.attributes) {
      if (attr.name.startsWith('data-') && attr.name !== 'data-viz') {
        const key = attr.name.slice(5);
        let val = attr.value;
        // simple type coercion
        if (val === 'true') val = true;
        else if (val === 'false') val = false;
        else if (!isNaN(parseFloat(val)) && isFinite(val)) val = parseFloat(val);
        props[key] = val;
      }
    }
    try {
      if (!window.VIZ_REGISTRY[name]) {
        // try to dynamically load
        await loadScript(`viz/${name}.js`);
      }
      const fn = window.VIZ_REGISTRY[name];
      if (typeof fn !== 'function') {
        el.innerHTML = `<div class="viz-error">Viz '${name}' not registered.</div>`;
        continue;
      }
      // clear and wrap
      const container = document.createElement('div');
      container.className = 'viz';
      el.replaceChildren(container);
      fn(container, props);
    } catch (e) {
      console.error('viz error', name, e);
      el.innerHTML = `<div class="viz-error">Error loading viz '${name}': ${e.message}</div>`;
    }
  }
};

function loadScript(src) {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) return resolve();
    const s = document.createElement('script');
    s.src = src;
    s.onload = resolve;
    s.onerror = () => reject(new Error(`Failed to load ${src}`));
    document.head.appendChild(s);
  });
}

// helper: lazy-load an ES module from CDN
window.loadModule = function (url) {
  return import(url);
};

// helper: ensure D3 is available
window.ensureD3 = async function () {
  if (window.d3) return window.d3;
  await loadScript('https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js');
  return window.d3;
};

// helper: ensure Three.js is available
window.ensureThree = async function () {
  if (window.THREE) return window.THREE;
  await loadScript('https://cdn.jsdelivr.net/npm/three@0.160/build/three.min.js');
  return window.THREE;
};
