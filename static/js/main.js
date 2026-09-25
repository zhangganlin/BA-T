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
