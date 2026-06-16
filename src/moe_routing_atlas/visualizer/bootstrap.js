/** Trace list loader — runs without Three.js so traces appear even if WebGL fails. */
(function () {
    /** Canonical short demo trace (10 tok) — always prefer over DESC list head. */
    const DEMO_TRACE_ID = 1;
    function apiBase() {
        if (window.location.protocol === 'file:') {
            return 'http://127.0.0.1:8777';
        }
        return window.location.origin;
    }

    function setStatus(msg, kind) {
        const el = document.getElementById('status');
        if (!el) return;
        el.textContent = msg;
        el.className = kind || '';
    }

    function traceId(t) {
        return t.id ?? t.trace_id;
    }

    function renderChip(t) {
        const id = traceId(t);
        const el = document.createElement('div');
        el.className = 'trace-chip';
        el.dataset.id = String(id);
        const experts = t.num_experts ?? '?';
        el.textContent = experts >= 128 ? `#${id}·${experts}E` : '#' + id;
        const model = t.model_name || t.model_id || 'trace';
        const topk = t.top_k ?? '?';
        el.title = `${model} · ${experts}E top-${topk} · ${t.num_tokens ?? '?'} tok`;
        if (t.num_experts >= 128) el.style.borderColor = '#3a7a9a';
        if (id === DEMO_TRACE_ID) {
            el.style.color = '#8ef';
            el.classList.add('demo-trace');
            el.textContent = `#${id} demo`;
        }
        el.onclick = () => requestTraceLoad(id);
        return el;
    }

    function requestTraceLoad(id) {
        setStatus('Loading trace #' + id + '…');
        window.__atlasPendingTraceId = id;
        window.dispatchEvent(new CustomEvent('atlas:load-trace', { detail: id }));
    }

    async function fetchWithTimeout(url, ms = 8000) {
        const ctrl = new AbortController();
        const timer = setTimeout(() => ctrl.abort(), ms);
        try {
            return await fetch(url, { signal: ctrl.signal });
        } finally {
            clearTimeout(timer);
        }
    }

    async function fetchTraces() {
        const grid = document.getElementById('traces');
        if (!grid) return;

        setStatus('Connecting to API…');

        try {
            const base = apiBase();
            const healthR = await fetchWithTimeout(base + '/health');
            const health = healthR.ok ? await healthR.json() : {};
            const [recentR, demoR] = await Promise.all([
                // Fetch up to 500 traces to enable rich selection on their 2k traces database
                fetchWithTimeout(base + '/traces?limit=500&order=desc'),
                fetchWithTimeout(base + '/traces?limit=1&order=asc'),
            ]);
            if (!recentR.ok) throw new Error('HTTP ' + recentR.status);
            const recent = await recentR.json();
            const demoRow = demoR.ok ? await demoR.json() : [];
            const demo = demoRow[0] ?? null;

            grid.innerHTML = '';
            if (!recent.length && !demo) {
                setStatus('No traces in database — run: moe-atlas dev', 'error');
                return;
            }

            const seen = new Set();
            const chips = [];
            if (demo) {
                chips.push(demo);
                seen.add(traceId(demo));
            }
            for (const t of recent) {
                const id = traceId(t);
                if (!seen.has(id)) {
                    chips.push(t);
                    seen.add(id);
                }
            }

            // Stateful Trace Selection Storage
            let allTraces = chips;
            let currentSearch = '';
            let currentFilter = 'all';

            // Premium metadata hover elements
            const previewCard = document.getElementById('trace-preview-card');
            const prevModel = document.getElementById('prev-model');
            const prevMeta = document.getElementById('prev-meta');
            const prevSnippet = document.getElementById('prev-snippet');

            function showPreview(t) {
                if (!previewCard) return;
                const id = traceId(t);
                const modelName = t.model_name || t.model_id || 'UNKNOWN MODEL';
                const tokenCount = t.num_tokens ?? '?';
                const experts = t.num_experts ?? '?';
                const top_k = t.top_k ?? '?';
                
                prevModel.textContent = modelName;
                prevMeta.textContent = `Trace #${id} · ${experts} Experts (top-${top_k}) · ${tokenCount} Tokens`;
                
                // Truncate snippet elegantly
                const clipText = t.text || t.text_preview || 'No prompt snippet available.';
                prevSnippet.textContent = clipText.length > 95 ? clipText.slice(0, 92) + '...' : clipText;
                
                previewCard.style.display = 'block';
            }

            function hidePreview() {
                // If active trace is selected, keep it as stable showcase otherwise revert
                const activeChipId = document.querySelector('.trace-chip.active')?.dataset.id;
                if (activeChipId) {
                    const activeTrace = allTraces.find(t => String(traceId(t)) === activeChipId);
                    if (activeTrace) {
                        showPreview(activeTrace);
                        return;
                    }
                }
                if (previewCard) previewCard.style.display = 'none';
            }

            function createChipElement(t) {
                const id = traceId(t);
                const el = document.createElement('div');
                el.className = 'trace-chip';
                el.dataset.id = String(id);
                const experts = t.num_experts ?? '?';
                el.textContent = experts >= 128 ? `#${id}·${experts}E` : '#' + id;
                
                if (id === DEMO_TRACE_ID) {
                    el.style.color = '#8ef';
                    el.classList.add('demo-trace');
                    el.textContent = `#${id} demo`;
                }

                // Bind hover events for preview card showcase
                el.addEventListener('mouseenter', () => showPreview(t));
                el.addEventListener('mouseleave', () => hidePreview());

                el.onclick = () => {
                    requestTraceLoad(id);
                    // Select
                    document.querySelectorAll('.trace-chip').forEach(c => c.classList.toggle('active', c.dataset.id === String(id)));
                    showPreview(t);
                };
                return el;
            }

            const collapsedGroups = new Set();

            function renderFilteredTraces() {
                grid.innerHTML = '';
                
                // 1. Filter
                let filtered = allTraces.filter(t => {
                    // Model filter
                    const mName = (t.model_name || t.model_id || '').toLowerCase();
                    if (currentFilter !== 'all' && !mName.includes(currentFilter.toLowerCase())) {
                        return false;
                    }
                    // Search term
                    if (currentSearch.trim()) {
                        const term = currentSearch.toLowerCase().trim();
                        const id = String(traceId(t)).toLowerCase();
                        const text = (t.text || t.text_preview || '').toLowerCase();
                        if (!id.includes(term) && !text.includes(term)) {
                            return false;
                        }
                    }
                    return true;
                });

                if (filtered.length === 0) {
                    const none = document.createElement('div');
                    none.style.color = '#555';
                    none.style.fontSize = '9px';
                    none.style.textAlign = 'center';
                    none.style.gridColumn = '1/span 5';
                    none.style.padding = '20px 0';
                    none.textContent = 'No matching traces found';
                    grid.appendChild(none);
                    return;
                }

                // 2. Group by model type dynamically!
                const groups = {};
                filtered.forEach(t => {
                    let mName = t.model_name || t.model_id || 'Other Traces';
                    // clean model tags dynamically for visual grouping headers
                    if (mName.includes('Qwen3.6')) mName = 'Qwen 3.6 Traces';
                    else if (mName.includes('Qwen3.5')) mName = 'Qwen 3.5 Traces';
                    else if (mName.includes('Qwen1.5')) mName = 'Qwen 1.5 Traces';
                    else if (traceId(t) === DEMO_TRACE_ID) mName = 'Demo Traces';
                    
                    if (!groups[mName]) groups[mName] = [];
                    groups[mName].push(t);
                });

                // 3. Render Groups with Collapse/Expand features
                Object.keys(groups).forEach(key => {
                    const header = document.createElement('div');
                    header.className = 'trace-group-header';
                    if (collapsedGroups.has(key)) {
                        header.classList.add('collapsed');
                    }
                    header.innerHTML = `<span>${key} (${groups[key].length})</span>`;
                    header.onclick = () => {
                        if (collapsedGroups.has(key)) {
                            collapsedGroups.delete(key);
                        } else {
                            collapsedGroups.add(key);
                        }
                        renderFilteredTraces();
                        // Re-establish active status
                        const activeId = window.__atlasPendingTraceId;
                        if (activeId) {
                            document.querySelectorAll('.trace-chip').forEach(c => c.classList.toggle('active', c.dataset.id === String(activeId)));
                        }
                    };
                    grid.appendChild(header);

                    const contentWrap = document.createElement('div');
                    contentWrap.className = 'trace-group-content';
                    contentWrap.style.gridColumn = '1 / span 5';
                    
                    if (collapsedGroups.has(key)) {
                        contentWrap.classList.add('collapsed');
                    } else {
                        // Create grid inner content
                        const subGrid = document.createElement('div');
                        subGrid.className = 'trace-grid';
                        subGrid.style.maxHeight = 'none'; // Unlimit grouped sub-grids to avoid nested scrollbars
                        groups[key].forEach(t => {
                            subGrid.appendChild(createChipElement(t));
                        });
                        contentWrap.appendChild(subGrid);
                    }
                    grid.appendChild(contentWrap);
                });
            }

            // Wire UI Event Listeners
            const searchInput = document.getElementById('trace-search');
            if (searchInput) {
                searchInput.addEventListener('input', (e) => {
                    currentSearch = e.target.value;
                    renderFilteredTraces();
                });
            }

            const filterChips = document.querySelectorAll('#model-filters .filter-chip');
            filterChips.forEach(chip => {
                chip.onclick = () => {
                    filterChips.forEach(c => c.classList.remove('active'));
                    chip.classList.add('active');
                    currentFilter = chip.dataset.model || 'all';
                    renderFilteredTraces();
                };
            });

            // Initial Group rendering
            renderFilteredTraces();

            const dbName = (health.db || '').replace(/^.*[\\/]/, '');
            const total = health.traces ?? chips.length;
            const demoMeta = chips.find((t) => traceId(t) === DEMO_TRACE_ID) ?? demo ?? recent[0];
            const defaultId = chips.some((t) => traceId(t) === DEMO_TRACE_ID)
                ? DEMO_TRACE_ID
                : (demo ? traceId(demo) : traceId(recent[0]));
                
            const archHint = demoMeta?.num_experts
                ? ` · ${demoMeta.num_experts}E top-${demoMeta.top_k ?? '?'}`
                : '';
                
            setStatus(
                `${total} traces · ${dbName || 'db'}${archHint} — loaded system`,
                total >= 100 ? 'ok' : '',
            );
            
            // Auto click and highlight default trace inside preview
            requestTraceLoad(defaultId);
            setTimeout(() => {
                document.querySelectorAll('.trace-chip').forEach(c => {
                    c.classList.toggle('active', c.dataset.id === String(defaultId));
                });
                const defaultTrace = allTraces.find(t => traceId(t) === defaultId);
                if (defaultTrace) showPreview(defaultTrace);
            }, 250);

        } catch (e) {
            const hint = window.location.protocol === 'file:'
                ? 'Open http://127.0.0.1:8777/visualizer/'
                : 'Run: moe-atlas dev';
            const msg = e?.name === 'AbortError' ? 'Request timed out' : (e?.message || String(e));
            setStatus('Cannot load traces: ' + msg + '. ' + hint, 'error');
            grid.textContent = 'No connection';
            console.error('[atlas] trace list failed:', e);
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', fetchTraces);
    } else {
        fetchTraces();
    }
})();