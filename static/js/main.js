// Highlight best / second-best among two-view methods in the main table.
(function highlightMainTable() {
  var table = document.getElementById('table-main');
  if (!table) return;
  var rows = Array.prototype.slice.call(table.querySelectorAll('tbody tr.tv'));
  // Column index (1-based, after the method name) -> true if higher is better.
  var higherBetter = { 1: 1, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0, 7: 1, 8: 1, 9: 1, 10: 1, 11: 0, 12: 0, 13: 0, 14: 1 };
  Object.keys(higherBetter).forEach(function (c) {
    var hb = higherBetter[c];
    var vals = rows.map(function (r) { return parseFloat(r.cells[c].textContent); });
    var uniq = vals.filter(function (v, i) { return vals.indexOf(v) === i; })
      .sort(function (a, b) { return hb ? b - a : a - b; });
    rows.forEach(function (r, i) {
      if (vals[i] === uniq[0]) r.cells[c].classList.add('best');
      else if (vals[i] === uniq[1]) r.cells[c].classList.add('second');
    });
  });
})();

// Convergence charts from Table 2 (7Scenes).
(function drawCharts() {
  var host = document.getElementById('charts');
  if (!host) return;
  var iters = [1, 2, 3, 4];
  var charts = [
    { title: 'Translation AUC@0.05m ↑', bat: [0.322, 0.401, 0.421, 0.422], base: [0.101, 0.133, 0.146, 0.151], min: 0, max: 0.48, ticks: 4, fmt: 3 },
    { title: '3D correspondence error (m) ↓', bat: [0.046, 0.032, 0.028, 0.026], base: [0.062, 0.048, 0.045, 0.045], min: 0.02, max: 0.07, ticks: 5, fmt: 3 }
  ];
  var W = 420, H = 230, m = { l: 44, r: 118, t: 14, b: 30 };
  var NS = 'http://www.w3.org/2000/svg';
  function el(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  charts.forEach(function (c) {
    var box = document.createElement('div');
    box.className = 'chart';
    var h = document.createElement('h5');
    h.textContent = c.title;
    box.appendChild(h);
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': c.title });
    var x = function (i) { return m.l + (i - 1) * (W - m.l - m.r) / 3; };
    var y = function (v) { return m.t + (1 - (v - c.min) / (c.max - c.min)) * (H - m.t - m.b); };
    var grid = el('g', { class: 'grid' }, svg), axis = el('g', { class: 'axis' }, svg);
    for (var t = 0; t <= c.ticks; t++) {
      var v = c.min + t * (c.max - c.min) / c.ticks;
      el('line', { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }, grid);
      el('text', { x: m.l - 8, y: y(v) + 4, 'text-anchor': 'end' }, axis).textContent = v.toFixed(2);
    }
    iters.forEach(function (i) {
      el('text', { x: x(i), y: H - m.b + 18, 'text-anchor': 'middle' }, axis).textContent = i;
    });
    el('text', { x: (m.l + W - m.r) / 2, y: H - 1, 'text-anchor': 'middle' }, axis).textContent = 'iteration';
    [['base', 'var(--base)', 'ViSTA† iter.'], ['bat', 'var(--bat)', 'BA-T']].forEach(function (s) {
      var d = c[s[0]].map(function (v, k) { return (k ? 'L' : 'M') + x(iters[k]) + ' ' + y(v); }).join(' ');
      el('path', { d: d, fill: 'none', stroke: s[1], 'stroke-width': 2.5, 'stroke-linejoin': 'round' }, svg);
      c[s[0]].forEach(function (v, k) {
        var dot = el('circle', { cx: x(iters[k]), cy: y(v), r: 4, fill: '#fff', stroke: s[1], 'stroke-width': 2 }, svg);
        el('title', {}, dot).textContent = s[2] + ', iter ' + iters[k] + ': ' + v.toFixed(c.fmt);
      });
      var last = c[s[0]][3];
      var lbl = el('text', { x: x(4) + 10, y: y(last) + 4, class: 'lbl-end', fill: s[1] }, svg);
      lbl.textContent = s[2] + ' ' + last.toFixed(c.fmt);
    });
    box.appendChild(svg);
    host.appendChild(box);
  });
})();

// Copy BibTeX.
(function copyBib() {
  var btn = document.getElementById('copy-bib');
  if (!btn) return;
  btn.addEventListener('click', function () {
    var text = document.getElementById('bib-text').textContent;
    var done = function () { btn.textContent = 'Copied'; setTimeout(function () { btn.textContent = 'Copy'; }, 1500); };
    if (navigator.clipboard) navigator.clipboard.writeText(text).then(done, function () {});
    else {
      var r = document.createRange(); r.selectNodeContents(document.getElementById('bib-text'));
      var sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(r);
      try { document.execCommand('copy'); done(); } catch (e) {}
    }
  });
})();

// Only play demo videos while visible (saves CPU on long pages).
(function lazyVideos() {
  if (!('IntersectionObserver' in window)) return;
  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      var v = e.target;
      if (e.isIntersecting) { if (v.preload === 'none') v.preload = 'auto'; var p = v.play(); if (p) p.catch(function () {}); }
      else v.pause();
    });
  }, { rootMargin: '100px' });
  document.querySelectorAll('video').forEach(function (v) { io.observe(v); });
})();
