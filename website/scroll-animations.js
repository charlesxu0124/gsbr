(() => {
  'use strict';
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');

  // All automatic playback shares the same viewport and background-tab gate.
  function observe(target, onVisibility) {
    let inView = false;
    const sync = () => onVisibility(inView && !document.hidden);
    const observer = new IntersectionObserver(entries => {
      const entry = entries[entries.length - 1];
      inView = entry.isIntersecting && entry.intersectionRatio >= .15;
      target.dataset.inView = String(inView);
      sync();
    }, { threshold: [0, .15], rootMargin: '-78px 0px -32px 0px' });
    observer.observe(target);
    document.addEventListener('visibilitychange', sync);
    return () => {
      observer.disconnect();
      document.removeEventListener('visibilitychange', sync);
    };
  }
  window.GSBRMotion = { observe };

  document.querySelectorAll('.benchmarks video').forEach(video => {
    let visible = false;
    video.autoplay = false;
    video.muted = true;
    const sync = () => {
      if (visible && !reduced.matches) {
        const play = video.play();
        if (play) play.then(() => {
          // A pending play request can resolve after a fast scroll away.
          if (!visible || reduced.matches) video.pause();
        }).catch(() => {});
      } else video.pause();
    };
    observe(video, value => { visible = value; sync(); });
    reduced.addEventListener('change', sync);
  });

  async function resultChart(root) {
    const response = await fetch('assets/throughput-animation.svg?v=bronze-name');
    if (!response.ok) throw Error('Result chart unavailable');
    const documentSVG = new DOMParser().parseFromString(await response.text(), 'image/svg+xml');
    if (documentSVG.querySelector('parsererror')) throw Error('Invalid result chart');
    const svg = document.importNode(documentSVG.documentElement, true);
    const baseline = Number(svg.dataset.baseline);
    const parts = [...svg.querySelectorAll('[data-result-part]')];
    if (parts.filter(part => part.dataset.resultPart === 'bar').length !== 40) throw Error('Incomplete result chart');
    const duration = 1654;
    let elapsed = reduced.matches ? duration : 0;
    let visible = false, frame = 0, last = null;

    function draw() {
      for (const part of parts) {
        const index = Number(part.dataset.barIndex);
        const delay = Math.floor(index / 8) * 100 + (index % 8) * 22;
        const progress = Math.max(0, Math.min(1, (elapsed - delay) / 1100));
        const height = 1 - Math.pow(1 - progress, 3);
        part.setAttribute('transform', `translate(0 ${baseline * (1 - height)}) scale(1 ${height})`);
        // Keep error-bar caps invisible until their corresponding bar has begun.
        part.style.opacity = String(progress === 0 ? 0 : 1);
      }
      root.dataset.progress = String(Math.min(1, elapsed / duration));
      root.dataset.animationState = elapsed >= duration ? 'complete' : visible ? 'playing' : 'waiting';
    }
    function tick(now) {
      frame = 0;
      if (!visible || document.hidden) { last = null; return; }
      if (last !== null) elapsed = Math.min(duration, elapsed + now - last);
      last = now;
      draw();
      sync();
    }
    function sync() {
      if (visible && !document.hidden && elapsed < duration && !reduced.matches) {
        if (!frame) frame = requestAnimationFrame(tick);
      } else {
        if (frame) cancelAnimationFrame(frame);
        frame = 0;
        last = null;
      }
    }
    draw();
    root.replaceChildren(svg);
    observe(root, value => { visible = value; sync(); });
    reduced.addEventListener('change', () => {
      if (reduced.matches) { elapsed = duration; draw(); }
      sync();
    });
  }
  document.querySelectorAll('[data-result-chart]').forEach(root => {
    resultChart(root).catch(error => console.warn(error.message));
  });
  // Reveal only the paper's data layers; axes, ticks, and labels stay fixed.
  document.querySelectorAll('[data-training-curve]').forEach(root => {
    const baselines = root.querySelector('[data-curve-reveal="baselines"]');
    const gsbr = root.querySelector('[data-curve-reveal="gsbr"]');
    const [x, y, width, height] = root.dataset.plotBounds.split(' ').map(Number);
    if (!baselines || !gsbr || ![x, y, width, height].every(Number.isFinite)) return;
    const baselineDuration = 1100, gsbrDuration = 1000;
    const duration = baselineDuration + gsbrDuration;
    let elapsed = reduced.matches ? duration : 0;
    let visible = false, frame = 0, last = null;

    function draw() {
      const progress = Math.min(1, elapsed / duration);
      const baselineProgress = Math.min(1, elapsed / baselineDuration);
      const gsbrProgress = Math.max(0, Math.min(1, (elapsed - baselineDuration) / gsbrDuration));
      baselines.setAttribute('width', String(width * baselineProgress));
      gsbr.setAttribute('width', String(width * gsbrProgress));
      root.dataset.baselineProgress = String(baselineProgress);
      root.dataset.gsbrProgress = String(gsbrProgress);
      root.dataset.animationPhase = progress === 1 ? 'complete' : elapsed < baselineDuration ? 'baselines' : 'gsbr';
      root.dataset.progress = String(progress);
      root.dataset.animationState = progress === 1 ? 'complete' : visible ? 'playing' : 'waiting';
    }
    function tick(now) {
      frame = 0;
      if (!visible || document.hidden) { last = null; return; }
      if (last !== null) elapsed = Math.min(duration, elapsed + now - last);
      last = now;
      draw();
      sync();
    }
    function sync() {
      if (visible && !document.hidden && elapsed < duration && !reduced.matches) {
        if (!frame) frame = requestAnimationFrame(tick);
      } else {
        if (frame) cancelAnimationFrame(frame);
        frame = 0;
        last = null;
      }
      root.dataset.animationState = elapsed >= duration ? 'complete' : visible ? 'playing' : 'waiting';
    }
    draw();
    observe(root, value => { visible = value; sync(); });
    reduced.addEventListener('change', () => {
      if (reduced.matches) { elapsed = duration; draw(); }
      sync();
    });
  });
})();
