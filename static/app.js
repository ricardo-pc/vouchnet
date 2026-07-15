/* VouchNet — the trust graph.
 *
 * No framework, no build step, no CDN: a force-directed layout and canvas
 * renderer written out longhand, served as a static file by the same FastAPI
 * container as the API. That is a deliberate trade. A React + d3 bundle would
 * mean a node toolchain, a build step on deploy, and ~200KB over the wire, to
 * draw one graph of ~60 nodes. The physics below is about 60 lines.
 *
 * The graph is the product: every node is an agent, every edge a review, and
 * the colour is the trust score the backend computed. Nothing here is
 * decorative-only — if it moves, it is showing you something in the data.
 */
(() => {
  'use strict';

  const canvas = document.getElementById('graph');
  const ctx = canvas.getContext('2d');

  const DIMENSIONS = ['accuracy', 'speed', 'reliability', 'clarity', 'safety'];

  const state = {
    mode: 'ledger',
    seed: 7,
    attacks: [],
    data: null,
    nodes: [],
    links: [],
    byId: new Map(),
    selected: null,
    hover: null,
    view: { x: 0, y: 0, k: 1 },
    alpha: 1,
    pulses: [],
    tickerX: 0,
    dragging: null,
    panning: null,
    busy: false,
  };

  /* ------------------------------------------------------------ helpers */

  const $ = (id) => document.getElementById(id);
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const fmt = (v, d = 2) => (v == null ? '—' : Number(v).toFixed(d));

  // Deterministic jitter from the node id, so a reload lays the graph out the
  // same way instead of reshuffling the whole picture in front of the user.
  function hash(str) {
    let h = 2166136261;
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return (h >>> 0) / 4294967295;
  }

  function colorFor(node) {
    if (node.risk >= 0.5) return '#ef4444';
    const s = node.trust_stars;
    if (s == null) return '#5a5e70';
    if (s >= 4.2) return '#4ade80';
    if (s >= 3.4) return '#d99b0c';
    if (s >= 2.6) return '#f97316';
    return '#ef4444';
  }

  const radiusFor = (node) => 4.5 + Math.sqrt(node.reviews || 0) * 2.3;

  function starString(n) {
    return '★'.repeat(Math.round(n)) + '☆'.repeat(5 - Math.round(n));
  }

  /* ---------------------------------------------------------- data load */

  async function load(reheat = true) {
    state.busy = true;
    $('loading').hidden = false;
    try {
      let data;
      if (state.mode === 'sandbox') {
        const res = await fetch('/sandbox/world', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ seed: state.seed, attacks: state.attacks }),
        });
        if (!res.ok) throw new Error(`sandbox ${res.status}`);
        data = await res.json();
      } else {
        const res = await fetch('/graph');
        if (!res.ok) throw new Error(`graph ${res.status}`);
        data = await res.json();
      }
      state.data = data;
      sync(data, reheat);
      renderBoard();
      renderStats();
      renderTicker();
      renderEmpty();
      if (state.selected) {
        const fresh = state.byId.get(state.selected.id);
        if (fresh) selectNode(fresh, false);
        else clearSelection();
      }
    } catch (err) {
      $('loading').innerHTML = `<span style="color:#ef4444">could not load the graph: ${err.message}</span>`;
      return;
    } finally {
      state.busy = false;
    }
    $('loading').hidden = true;
  }

  // Merge new data into the running simulation, keeping the positions of nodes
  // that already exist. Rebuilding from scratch would teleport every agent on
  // each attack, and the whole point is to watch the ring *arrive*.
  function sync(data, reheat) {
    const previous = state.byId;
    const nodes = [];
    const byId = new Map();
    const w = canvas.clientWidth || 1200;
    const h = canvas.clientHeight || 800;

    for (const raw of data.nodes) {
      const old = previous.get(raw.id);
      const angle = hash(raw.id) * Math.PI * 2;
      const spread = 90 + hash(raw.id + 'r') * 220;
      const node = Object.assign({}, raw, {
        x: old ? old.x : w / 2 + Math.cos(angle) * spread,
        y: old ? old.y : h / 2 + Math.sin(angle) * spread,
        vx: old ? old.vx : 0,
        vy: old ? old.vy : 0,
        born: old ? old.born : performance.now(),
      });
      nodes.push(node);
      byId.set(node.id, node);
    }

    state.nodes = nodes;
    state.byId = byId;
    state.links = data.links
      .map((l) => ({ source: byId.get(l.source), target: byId.get(l.target), stars: l.stars, attack: l.attack, comment: l.comment }))
      .filter((l) => l.source && l.target);

    for (const node of nodes) node.degree = 0;
    for (const link of state.links) { link.source.degree++; link.target.degree++; }

    state.pulses = [];
    if (reheat) state.alpha = 1;
  }

  /* ------------------------------------------------------------ physics */

  function tick() {
    const nodes = state.nodes;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    const cx = w / 2;
    const cy = h / 2;
    const a = state.alpha;

    // Repulsion — every pair pushes apart, so nodes do not stack. O(n²), which
    // at n≈60 is ~1800 pairs per frame: far cheaper than a quadtree's
    // bookkeeping at this size.
    for (let i = 0; i < nodes.length; i++) {
      const p = nodes[i];
      for (let j = i + 1; j < nodes.length; j++) {
        const q = nodes[j];
        let dx = q.x - p.x;
        let dy = q.y - p.y;
        let d2 = dx * dx + dy * dy;
        if (d2 < 1) { dx = (hash(p.id) - 0.5) * 2; dy = (hash(q.id) - 0.5) * 2; d2 = 4; }
        if (d2 > 250000) continue; // Beyond ~500px the force is noise.
        const d = Math.sqrt(d2);
        const force = (2800 * a) / d2;
        const fx = (dx / d) * force;
        const fy = (dy / d) * force;
        p.vx -= fx; p.vy -= fy;
        q.vx += fx; q.vy += fy;
      }
    }

    // Springs — a review pulls reviewer and reviewed together, so clusters of
    // agents that actually work with each other end up near each other. This
    // is why a collusion ring looks like a ring: it only links to itself.
    //
    // Each spring is divided by the degree of its busier endpoint. Without
    // that, a well-reviewed agent with 30 edges gets pulled 30 times per frame
    // and drags the whole network into one unreadable knot -- popularity would
    // decide the layout instead of structure.
    for (const link of state.links) {
      const dx = link.target.x - link.source.x;
      const dy = link.target.y - link.source.y;
      const d = Math.hypot(dx, dy) || 1;
      const degree = Math.max(link.source.degree, link.target.degree, 1);
      const force = ((d - 150) * 0.05 * a) / degree;
      const fx = (dx / d) * force;
      const fy = (dy / d) * force;
      link.source.vx += fx; link.source.vy += fy;
      link.target.vx -= fx; link.target.vy -= fy;
    }

    for (const node of nodes) {
      if (node === state.dragging) { node.vx = 0; node.vy = 0; continue; }
      // Gravity toward the middle keeps disconnected components on screen —
      // attackers have no edges to the honest network and would drift away.
      node.vx += (cx - node.x) * 0.0022 * a;
      node.vy += (cy - node.y) * 0.0022 * a;
      node.vx *= 0.86;
      node.vy *= 0.86;
      node.x += clamp(node.vx, -18, 18);
      node.y += clamp(node.vy, -18, 18);

      // Soft walls. An attack cluster links only to itself, so repulsion from
      // the honest network pushes it off-canvas -- exactly the thing the user
      // came to look at. Nudge it back rather than hard-clamping, which would
      // pin nodes to the edge and visibly break the physics.
      const pad = 54;
      if (node.x < pad) node.vx += (pad - node.x) * 0.06;
      if (node.x > w - pad) node.vx -= (node.x - (w - pad)) * 0.06;
      if (node.y < pad) node.vy += (pad - node.y) * 0.06;
      if (node.y > h - pad) node.vy -= (node.y - (h - pad)) * 0.06;
    }

    if (state.alpha > 0.02) state.alpha *= 0.985;
  }

  /* ----------------------------------------------------------- pulses */

  function spawnPulse() {
    if (!state.links.length || state.pulses.length > 26) return;
    const link = state.links[(Math.random() * state.links.length) | 0];
    state.pulses.push({ link, t: 0, speed: 0.006 + Math.random() * 0.006 });
  }

  /* ----------------------------------------------------------- render */

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = canvas.clientWidth * dpr;
    canvas.height = canvas.clientHeight * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function draw() {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    ctx.save();
    ctx.clearRect(0, 0, w, h);
    ctx.translate(state.view.x, state.view.y);
    ctx.scale(state.view.k, state.view.k);

    const focus = state.hover || state.selected;
    const near = new Set();
    if (focus) {
      near.add(focus.id);
      for (const l of state.links) {
        if (l.source.id === focus.id) near.add(l.target.id);
        if (l.target.id === focus.id) near.add(l.source.id);
      }
    }

    // Edges.
    for (const link of state.links) {
      const involved = focus && (link.source.id === focus.id || link.target.id === focus.id);
      const dim = focus && !involved;
      const positive = link.stars >= 4;
      let stroke;
      if (link.attack) stroke = `rgba(239,68,68,${dim ? 0.06 : 0.34})`;
      else if (positive) stroke = `rgba(217,155,12,${dim ? 0.035 : involved ? 0.5 : 0.14})`;
      else stroke = `rgba(255,255,255,${dim ? 0.02 : involved ? 0.22 : 0.06})`;
      ctx.strokeStyle = stroke;
      ctx.lineWidth = involved ? 1.4 : 0.7;
      ctx.beginPath();
      ctx.moveTo(link.source.x, link.source.y);
      ctx.lineTo(link.target.x, link.target.y);
      ctx.stroke();
    }

    // Pulses — a review travelling from reviewer to reviewed.
    for (const pulse of state.pulses) {
      const { source, target } = pulse.link;
      const x = source.x + (target.x - source.x) * pulse.t;
      const y = source.y + (target.y - source.y) * pulse.t;
      const fade = Math.sin(pulse.t * Math.PI);
      ctx.fillStyle = pulse.link.attack
        ? `rgba(239,68,68,${fade})`
        : `rgba(217,155,12,${fade * 0.85})`;
      ctx.beginPath();
      ctx.arc(x, y, 1.8, 0, Math.PI * 2);
      ctx.fill();
    }

    // Nodes.
    const now = performance.now();
    for (const node of state.nodes) {
      const r = radiusFor(node);
      const color = colorFor(node);
      const dim = focus && !near.has(node.id);
      const alpha = dim ? 0.22 : 1;
      const grow = clamp((now - node.born) / 500, 0, 1);
      const rr = r * (0.4 + 0.6 * grow);

      ctx.globalAlpha = alpha;

      // Glow.
      const glow = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, rr * 3.6);
      glow.addColorStop(0, color + '55');
      glow.addColorStop(1, color + '00');
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(node.x, node.y, rr * 3.6, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(node.x, node.y, rr, 0, Math.PI * 2);
      ctx.fill();

      // Seed agents wear a gold ring: they are where trust is injected.
      if (node.seed) {
        ctx.strokeStyle = '#d99b0c';
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.arc(node.x, node.y, rr + 4, 0, Math.PI * 2);
        ctx.stroke();
      }
      // Flagged agents get a pulsing red halo.
      if (node.risk >= 0.5) {
        const beat = 0.5 + 0.5 * Math.sin(now / 320);
        ctx.strokeStyle = `rgba(239,68,68,${0.25 + beat * 0.5})`;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.arc(node.x, node.y, rr + 5 + beat * 3, 0, Math.PI * 2);
        ctx.stroke();
      }
      if (state.selected && node.id === state.selected.id) {
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.arc(node.x, node.y, rr + 7, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Labels only where they can be read: big nodes, or whatever is focused.
      const label = r > 8 || (focus && near.has(node.id)) || state.view.k > 1.5;
      if (label) {
        ctx.globalAlpha = dim ? 0.2 : 0.92;
        ctx.fillStyle = '#e8e9ee';
        ctx.font = '10px ui-monospace, Menlo, monospace';
        ctx.textAlign = 'center';
        ctx.fillText(node.id, node.x, node.y + rr + 11);
      }
      ctx.globalAlpha = 1;
    }
    ctx.restore();
  }

  function frame() {
    tick();
    if (Math.random() < 0.22) spawnPulse();
    for (const pulse of state.pulses) pulse.t += pulse.speed;
    state.pulses = state.pulses.filter((p) => p.t < 1);
    draw();
    stepTicker();
    requestAnimationFrame(frame);
  }

  /* ------------------------------------------------------------ panels */

  function renderStats() {
    const s = state.data.stats;
    const mode = state.mode === 'sandbox' ? 'simulated' : 'on the ledger';
    $('stats').innerHTML =
      `<span><b>${s.agents}</b>agents</span>` +
      `<span><b>${s.reviews}</b>reviews ${mode}</span>` +
      `<span><b>${s.flagged}</b>flagged</span>`;
  }

  // An empty ledger is a legitimate state, not a failure: it means no agent
  // has filed a review yet. Say so, and point at the two ways forward.
  function renderEmpty() {
    const empty = state.nodes.length === 0;
    $('empty').hidden = !empty;
  }

  function renderBoard() {
    // Ranked by the lower edge of the credible interval, matching
    // GET /leaderboard: an unproven agent has a wide interval and sorts down.
    const rows = state.data.nodes
      .filter((n) => n.reviews > 0)
      .sort((a, b) => b.interval[0] - a.interval[0] || b.reviews - a.reviews);
    $('board-body').innerHTML = rows
      .map(
        (n, i) => `<tr data-agent="${n.id}" class="${n.risk >= 0.5 ? 'is-flagged' : ''} ${
          state.selected && state.selected.id === n.id ? 'is-selected' : ''
        }">
          <td class="rank">${i + 1}</td>
          <td class="agent" title="${n.id}">${n.id}</td>
          <td class="num raw">${fmt(n.naive)}</td>
          <td class="num trust">${fmt(n.trust_stars)}<span class="pm">±${fmt(
            (n.interval[1] - n.interval[0]) / 2, 1
          )}</span></td>
        </tr>`
      )
      .join('');

    const prior = state.data.prior;
    $('prior-note').innerHTML =
      `RAW is the plain star average. TRUST is a Beta posterior weighted by ` +
      `TrustRank, ± half its 90% interval. Ranked by the <em>lower</em> edge of ` +
      `that interval, so an unproven agent cannot top the chart. Prior fitted ` +
      `from this population: <b>${fmt(prior.mean_stars)}★</b> worth ` +
      `<b>${fmt(prior.strength, 1)}</b> reviews; measured dispersion makes one ` +
      `trusted review worth <b>${fmt(prior.kappa, 1)}</b> observations.`;

    for (const tr of $('board-body').querySelectorAll('tr')) {
      tr.onclick = () => {
        const node = state.byId.get(tr.dataset.agent);
        if (node) selectNode(node);
      };
    }
  }

  function pentagonSVG(node) {
    const R = 52, cx = 92, cy = 76;
    const point = (i, r) => {
      const a = -Math.PI / 2 + (i * 2 * Math.PI) / 5;
      return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
    };
    let out = '';
    for (let ring = 1; ring <= 5; ring++) {
      const pts = [...Array(5)].map((_, i) => point(i, (R * ring) / 5).map((v) => v.toFixed(1)).join(',')).join(' ');
      out += `<polygon points="${pts}" fill="none" stroke="rgba(255,255,255,0.07)" stroke-width="1"/>`;
    }
    const trusted = [], naive = [];
    DIMENSIONS.forEach((dim, i) => {
      const [ax, ay] = point(i, R);
      out += `<line x1="${cx}" y1="${cy}" x2="${ax.toFixed(1)}" y2="${ay.toFixed(1)}" stroke="rgba(255,255,255,0.07)"/>`;
      const d = (node.dimensions || {})[dim];
      const [lx, ly] = point(i, R + 15);
      if (d && d.trusted != null) {
        trusted.push(point(i, (R * d.trusted) / 5));
        naive.push(point(i, (R * d.naive) / 5));
        out += `<text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" text-anchor="middle" dominant-baseline="middle" font-size="7.5" font-family="ui-monospace,Menlo,monospace" fill="#d99b0c">${dim} ${d.trusted.toFixed(1)}</text>`;
      } else {
        // Unrated stays grey. A dimension nobody scored is unknown, not zero —
        // the rule the whole pentagon rests on.
        out += `<text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" text-anchor="middle" dominant-baseline="middle" font-size="7.5" font-family="ui-monospace,Menlo,monospace" fill="#4a4d5c">${dim}</text>`;
      }
    });
    if (naive.length >= 3) {
      out += `<polygon points="${naive.map((p) => p.map((v) => v.toFixed(1)).join(',')).join(' ')}" fill="none" stroke="rgba(255,255,255,0.22)" stroke-width="1" stroke-dasharray="2 2"/>`;
    }
    if (trusted.length >= 3) {
      out += `<polygon points="${trusted.map((p) => p.map((v) => v.toFixed(1)).join(',')).join(' ')}" fill="rgba(217,155,12,0.20)" stroke="#d99b0c" stroke-width="1.6"/>`;
    }
    for (const [x, y] of trusted) out += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="2.2" fill="#d99b0c"/>`;
    return `<svg class="pentagon" viewBox="0 0 184 152" width="184" height="152" role="img" aria-label="reputation pentagon">${out}</svg>`;
  }

  function renderAgent(node) {
    $('agent-name').textContent = node.id;
    $('agent-panel').classList.remove('is-empty');

    const reviews = state.links
      .filter((l) => l.target.id === node.id)
      .slice(-14)
      .reverse();

    const pct = (v) => ((clamp(v, 1, 5) - 1) / 4) * 100;
    const ci = node.interval || [1, 5];

    const flagged = node.risk >= 0.5;
    const warn = flagged
      ? `<div class="warn"><strong>⚠ risk ${node.risk.toFixed(2)}</strong>${node.reasons.join('; ')}</div>`
      : '';
    const victim = !flagged && node.reasons && node.reasons.length
      ? `<div class="warn" style="background:rgba(217,155,12,0.08);border-color:rgba(217,155,12,0.3);color:#f0d9a0"><strong>⚑ under attack</strong>${node.reasons.join('; ')}. Its trust score barely moved.</div>`
      : '';

    const truth = node.truth != null
      ? `<div class="row"><span class="k">true quality (sandbox)</span><span class="v" style="color:#4ade80">${fmt(node.truth)}★</span></div>`
      : '';

    $('agent-body').innerHTML = `
      <div class="headline-score">
        <span class="big">${fmt(node.trust_stars)}</span>
        <span class="of">/ 5 trust score</span>
      </div>
      <p class="sub-note">${node.reviews} review${node.reviews === 1 ? '' : 's'} · ${fmt(node.evidence, 1)} effective</p>

      <div class="ci">
        <div class="ci-track">
          <div class="ci-band" style="left:${pct(ci[0])}%;width:${pct(ci[1]) - pct(ci[0])}%"></div>
          <div class="ci-naive" style="left:${pct(node.naive)}%" title="plain average: ${fmt(node.naive)}"></div>
          <div class="ci-mark" style="left:${pct(node.trust_stars)}%" title="trust score: ${fmt(node.trust_stars)}"></div>
        </div>
        <div class="ci-scale"><span>1★</span><span>90% credible interval — grey mark = raw average</span><span>5★</span></div>
      </div>

      ${warn}${victim}

      ${pentagonSVG(node)}
      <p class="panel-note" style="text-align:center">Gold = trust-weighted · dashed = raw average · grey axis = not yet rated</p>

      <div class="rows">
        <div class="row"><span class="k">raw average</span><span class="v">${fmt(node.naive)}★</span></div>
        <div class="row"><span class="k">bayesian only</span><span class="v">${fmt(node.bayes)}★</span></div>
        <div class="row"><span class="k">trust-weighted</span><span class="v hi">${fmt(node.trust_stars)}★</span></div>
        ${truth}
        <div class="row"><span class="k">its weight as a reviewer</span><span class="v">×${fmt(node.weight)}</span></div>
        <div class="row"><span class="k">trustrank mass</span><span class="v">${node.trust === 0 ? '0 — unreachable' : node.trust.toFixed(4)}</span></div>
      </div>

      <div class="subhead">Reviews received</div>
      <ul class="reviews">
        ${reviews
          .map((l) => {
            const w = state.byId.get(l.source.id);
            return `<li>
              <span class="stars ${l.stars <= 2 ? 'bad' : ''}">${starString(l.stars)}</span>
              ${l.comment ? ` ${l.comment}` : ''}
              <span class="by">by ${l.source.id} <span class="w">· weight ×${fmt(w ? w.weight : 1)}</span></span>
            </li>`;
          })
          .join('') || '<li class="panel-note">No reviews yet.</li>'}
      </ul>`;
  }

  function selectNode(node, rerender = true) {
    state.selected = node;
    renderAgent(node);
    if (rerender) renderBoard();
    $('attack-target').textContent = node.id;
  }

  function clearSelection() {
    state.selected = null;
    $('agent-panel').classList.add('is-empty');
    $('agent-name').textContent = 'Select an agent';
    $('agent-body').innerHTML = '<p class="panel-note">Click any node to see its reputation pentagon, credible interval, and review history.</p>';
    $('attack-target').textContent = 'auto (a mid-ranked agent)';
    renderBoard();
  }

  /* ------------------------------------------------------------ ticker */

  function renderTicker() {
    const links = state.links.slice(-26).reverse();
    if (!links.length) {
      $('ticker').innerHTML = '<span>no reviews yet — POST /reviews to file the first one</span>';
      return;
    }
    const item = (l) =>
      `<span><span class="${l.stars <= 2 ? 'tbad' : 'tstar'}">${starString(l.stars)}</span> ` +
      `<b>${l.target.id}</b>${l.comment ? ` — ${l.comment}` : ''} · by ${l.source.id}</span>`;
    // Duplicated so the strip can loop without a visible seam.
    $('ticker').innerHTML = links.map(item).join('') + links.map(item).join('');
    state.tickerX = 0;
  }

  function stepTicker() {
    const track = $('ticker');
    const half = track.scrollWidth / 2;
    if (!half) return;
    state.tickerX -= 0.45;
    if (-state.tickerX >= half) state.tickerX = 0;
    track.style.transform = `translateX(${state.tickerX}px)`;
  }

  /* ------------------------------------------------------------ attacks */

  function defaultTarget() {
    const candidates = state.data.nodes
      .filter((n) => n.reviews >= 3 && !n.seed && !n.malicious && n.risk < 0.5)
      .sort((a, b) => Math.abs(a.trust_stars - 3.3) - Math.abs(b.trust_stars - 3.3));
    return candidates[0] || null;
  }

  async function runAttack(kind) {
    if (state.busy) return;
    const target = state.selected && !state.selected.malicious ? state.selected : defaultTarget();
    if (!target) return;

    // Snapshot before, so the verdict compares the same agent to itself.
    const before = { naive: target.naive, trust: target.trust_stars, id: target.id };
    state.attacks.push({ kind, target: target.id, size: 10 });
    await load(true);

    const after = state.byId.get(before.id);
    if (!after) return;
    showVerdict(kind, before, after);
    selectNode(after, true);
  }

  function showVerdict(kind, before, after) {
    const dNaive = after.naive - before.naive;
    const dTrust = after.trust_stars - before.trust;
    const blocked = Math.abs(dNaive) > 0.01
      ? clamp(1 - Math.abs(dTrust) / Math.abs(dNaive), 0, 1)
      : 0;
    const label = { collusion_ring: 'Collusion ring', sybil_boost: 'Sybil boost', review_bomb: 'Review bomb' }[kind];

    const el = $('verdict');
    el.hidden = false;
    el.innerHTML = `
      <div class="verdict-head">${label} → ${before.id}</div>
      <div class="verdict-row">
        <span class="lab">raw average</span>
        <span><span style="color:#5a5e70">${fmt(before.naive)}</span> → ${fmt(after.naive)}
        <span class="delta bad">${dNaive >= 0 ? '+' : ''}${fmt(dNaive)}</span></span>
      </div>
      <div class="verdict-row">
        <span class="lab">VouchNet trust</span>
        <span><span style="color:#5a5e70">${fmt(before.trust)}</span> → ${fmt(after.trust_stars)}
        <span class="delta good">${dTrust >= 0 ? '+' : ''}${fmt(dTrust)}</span></span>
      </div>
      <div class="verdict-tag">
        <b>${(blocked * 100).toFixed(0)}% of the attack absorbed.</b>
        The attackers hold no TrustRank mass — nobody credible ever vouched for
        them — so their reviews carry ×0.15 weight instead of ×1.
      </div>`;
  }

  /* ---------------------------------------------------------- interaction */

  function toGraph(event) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left - state.view.x) / state.view.k,
      y: (event.clientY - rect.top - state.view.y) / state.view.k,
    };
  }

  function nodeAt(pt) {
    for (let i = state.nodes.length - 1; i >= 0; i--) {
      const node = state.nodes[i];
      const r = radiusFor(node) + 6;
      if ((node.x - pt.x) ** 2 + (node.y - pt.y) ** 2 <= r * r) return node;
    }
    return null;
  }

  canvas.addEventListener('mousedown', (e) => {
    const pt = toGraph(e);
    const node = nodeAt(pt);
    if (node) {
      state.dragging = node;
      state.alpha = Math.max(state.alpha, 0.35);
    } else {
      state.panning = { x: e.clientX - state.view.x, y: e.clientY - state.view.y };
      canvas.classList.add('is-dragging');
    }
  });

  window.addEventListener('mousemove', (e) => {
    if (state.dragging) {
      const pt = toGraph(e);
      state.dragging.x = pt.x;
      state.dragging.y = pt.y;
      return;
    }
    if (state.panning) {
      state.view.x = e.clientX - state.panning.x;
      state.view.y = e.clientY - state.panning.y;
      return;
    }
    const pt = toGraph(e);
    const node = nodeAt(pt);
    state.hover = node;
    canvas.style.cursor = node ? 'pointer' : 'grab';
    canvas.title = node
      ? `${node.id}\ntrust ${fmt(node.trust_stars)}★ · raw ${fmt(node.naive)}★ · ${node.reviews} reviews${node.risk >= 0.5 ? '\n⚠ ' + node.reasons.join('; ') : ''}`
      : '';
  });

  window.addEventListener('mouseup', (e) => {
    if (state.dragging) {
      // A click is a drag that never moved.
      const pt = toGraph(e);
      if (Math.hypot(state.dragging.x - pt.x, state.dragging.y - pt.y) < 4) selectNode(state.dragging);
    }
    state.dragging = null;
    state.panning = null;
    canvas.classList.remove('is-dragging');
  });

  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const k = clamp(state.view.k * (e.deltaY < 0 ? 1.12 : 0.89), 0.35, 3.5);
    // Zoom about the cursor rather than the origin.
    state.view.x = mx - ((mx - state.view.x) * k) / state.view.k;
    state.view.y = my - ((my - state.view.y) * k) / state.view.k;
    state.view.k = k;
  }, { passive: false });

  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') clearSelection();
  });

  /* -------------------------------------------------------------- wiring */

  for (const btn of document.querySelectorAll('[data-intro]')) {
    btn.onclick = async () => {
      setMode(btn.dataset.intro, false);
      $('intro').classList.add('is-gone');
      await load();
    };
  }

  for (const btn of document.querySelectorAll('.mode')) {
    btn.onclick = () => setMode(btn.dataset.mode, true);
  }

  function setMode(mode, reload) {
    state.mode = mode;
    for (const btn of document.querySelectorAll('.mode')) {
      const active = btn.dataset.mode === mode;
      btn.classList.toggle('is-active', active);
      btn.setAttribute('aria-selected', String(active));
    }
    $('attack-panel').hidden = mode !== 'sandbox';
    document.body.classList.toggle('mode-sandbox', mode === 'sandbox');
    $('verdict').hidden = true;
    state.attacks = [];
    clearSelection();
    if (reload) load();
  }

  for (const btn of document.querySelectorAll('[data-attack]')) {
    btn.onclick = () => runAttack(btn.dataset.attack);
  }

  $('reset-sandbox').onclick = () => {
    state.attacks = [];
    $('verdict').hidden = true;
    clearSelection();
    load();
  };

  $('close-agent').onclick = clearSelection;

  for (const btn of document.querySelectorAll('[data-collapse]')) {
    btn.onclick = () => $(btn.dataset.collapse).classList.toggle('is-collapsed');
  }

  window.addEventListener('resize', resize);

  /* ---------------------------------------------------------------- boot */

  resize();
  requestAnimationFrame(frame);

  // Deep links: /profile/{name} redirects here, and the old URLs still work.
  const wanted = new URLSearchParams(location.search).get('agent');
  if (wanted) {
    $('intro').classList.add('is-gone');
    load().then(() => {
      const node = state.byId.get(wanted);
      if (node) selectNode(node);
    });
  } else {
    load();
  }
})();
