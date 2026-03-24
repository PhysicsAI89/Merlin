/*
merlin 5.3 - ensemble frontend
shows individual model votes, combined prediction with confidence bands
*/

let currentTicker = '';
let priceChart = null;
let predictionChart = null;
let backtestChart = null;
let fullChartData = null;

const MODEL_COLOURS = ['#4a7cff', '#f0a030', '#a07cff'];

document.getElementById('ticker-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') fetchData();
});
document.querySelectorAll('.chart-range-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        document.querySelectorAll('.chart-range-btn').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
        if (fullChartData) renderChart(this.dataset.range);
    });
});

function showStatus(msg, type = '') {
    const bar = document.getElementById('status-bar');
    bar.className = 'status-bar ' + type;
    document.getElementById('status-text').textContent = msg;
    bar.classList.remove('hidden');
    if (type === 'success') setTimeout(() => bar.classList.add('hidden'), 4000);
}

async function fetchData() {
    const ticker = document.getElementById('ticker-input').value.trim().toUpperCase();
    if (!ticker) { showStatus('enter a ticker symbol first', 'error'); return; }
    const btn = document.getElementById('fetch-btn');
    btn.querySelector('.btn-text').textContent = 'fetching...';
    btn.querySelector('.btn-loader').classList.remove('hidden');
    btn.disabled = true;

    try {
        showStatus(`fetching data for ${ticker}...`);
        const res = await fetch('/api/fetch_data', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker })
        });
        const data = await res.json();
        if (!res.ok) { showStatus(data.error || 'failed', 'error'); return; }

        currentTicker = data.ticker;
        fullChartData = data.chart_data;

        document.getElementById('info-ticker').textContent = data.ticker;
        document.getElementById('info-price').textContent = '$' + data.current_price.toFixed(2);
        const ce = document.getElementById('info-change');
        const s = data.change >= 0 ? '+' : '';
        ce.textContent = `${s}$${data.change.toFixed(2)} (${s}${data.change_pct}%)`;
        ce.className = 'info-value ' + (data.change >= 0 ? 'positive' : 'negative');
        document.getElementById('info-points').textContent = data.data_points.toLocaleString();

        document.getElementById('stock-info').classList.remove('hidden');
        document.getElementById('chart-section').classList.remove('hidden');
        document.getElementById('train-section').classList.remove('hidden');
        document.getElementById('predict-section').classList.add('hidden');
        document.getElementById('backtest-section').classList.add('hidden');
        document.getElementById('prediction-results').classList.add('hidden');

        if (data.news && data.news.articles && data.news.articles.length > 0) {
            renderNews(data.news);
            document.getElementById('news-section').classList.remove('hidden');
        } else {
            document.getElementById('news-section').classList.add('hidden');
        }

        document.querySelector('.chart-range-btn.active').click();
        showStatus(`loaded ${data.data_points} data points for ${data.ticker}`, 'success');
    } catch (err) {
        showStatus('network error: ' + err.message, 'error');
    } finally {
        btn.querySelector('.btn-text').textContent = 'fetch data';
        btn.querySelector('.btn-loader').classList.add('hidden');
        btn.disabled = false;
    }
}

function renderNews(news) {
    const badge = document.getElementById('news-badge');
    let bc = 'neutral', bt = 'neutral';
    if (news.overall_score > 0.3) { bc = 'positive'; bt = 'positive'; }
    else if (news.overall_score < -0.3) { bc = 'negative'; bt = 'negative'; }
    badge.className = 'news-badge ' + bc;
    badge.textContent = bt + ' (' + (news.overall_score >= 0 ? '+' : '') + news.overall_score.toFixed(1) + ')';

    document.getElementById('news-list').innerHTML = news.articles.map(a => `
        <div class="news-item">
            <div class="news-sentiment-dot ${a.sentiment}"></div>
            <div class="news-content">
                <div class="news-title">${a.link ? `<a href="${a.link}" target="_blank" rel="noopener">${a.title}</a>` : a.title}</div>
                <div class="news-meta">${a.publisher}${a.date ? ' &middot; ' + a.date : ''}</div>
            </div>
        </div>
    `).join('');
}

function renderChart(range) {
    if (!fullChartData) return;
    let dates = fullChartData.dates, prices = fullChartData.close;
    let sma20 = fullChartData.sma_20, sma50 = fullChartData.sma_50;
    const total = dates.length;
    let start = 0;
    switch(range) {
        case '3m': start = Math.max(0, total - 66); break;
        case '6m': start = Math.max(0, total - 132); break;
        case '1y': start = Math.max(0, total - 252); break;
        case 'all': start = 0; break;
    }
    dates = dates.slice(start); prices = prices.slice(start);
    sma20 = sma20 ? sma20.slice(start) : null;
    sma50 = sma50 ? sma50.slice(start) : null;

    const ctx = document.getElementById('price-chart').getContext('2d');
    if (priceChart) priceChart.destroy();
    const grad = ctx.createLinearGradient(0, 0, 0, 300);
    grad.addColorStop(0, 'rgba(74,124,255,0.25)');
    grad.addColorStop(1, 'rgba(74,124,255,0.0)');

    const ds = [{ label: currentTicker + ' close', data: prices, borderColor: '#4a7cff', backgroundColor: grad, borderWidth: 1.5, fill: true, pointRadius: 0, tension: 0.1 }];
    if (sma20) ds.push({ label: 'SMA 20', data: sma20, borderColor: 'rgba(240,160,48,0.6)', borderWidth: 1, pointRadius: 0, fill: false, tension: 0.1, borderDash: [4,3] });
    if (sma50) ds.push({ label: 'SMA 50', data: sma50, borderColor: 'rgba(45,212,160,0.5)', borderWidth: 1, pointRadius: 0, fill: false, tension: 0.1, borderDash: [6,4] });

    priceChart = new Chart(ctx, {
        type: 'line', data: { labels: dates, datasets: ds },
        options: {
            responsive: true, maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            plugins: {
                legend: { labels: { font: { family: 'JetBrains Mono', size: 10 }, color: '#8888a0', boxWidth: 12 } },
                tooltip: { backgroundColor: '#16161f', borderColor: '#2a2a3a', borderWidth: 1, titleFont: { family: 'JetBrains Mono', size: 11 }, bodyFont: { family: 'JetBrains Mono', size: 12 }, titleColor: '#8888a0', bodyColor: '#e8e8f0', displayColors: false, callbacks: { label: c => c.parsed.y != null ? c.dataset.label + ': $' + c.parsed.y.toFixed(2) : '' } }
            },
            scales: {
                x: { grid: { color: 'rgba(42,42,58,0.3)' }, ticks: { font: { family: 'JetBrains Mono', size: 10 }, color: '#55556a', maxTicksLimit: 8 } },
                y: { grid: { color: 'rgba(42,42,58,0.3)' }, ticks: { font: { family: 'JetBrains Mono', size: 10 }, color: '#55556a', callback: v => '$' + v.toFixed(0) } }
            }
        }
    });
}

async function trainModel() {
    if (!currentTicker) { showStatus('fetch data first', 'error'); return; }
    const btn = document.getElementById('train-btn');
    const epochs = parseInt(document.getElementById('epochs-input').value) || 50;
    btn.querySelector('.btn-text').textContent = 'training...';
    btn.querySelector('.btn-loader').classList.remove('hidden');
    btn.disabled = true;
    document.getElementById('training-progress').classList.remove('hidden');
    document.getElementById('backtest-section').classList.add('hidden');

    try {
        const res = await fetch('/api/train', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ticker: currentTicker, epochs }) });
        const data = await res.json();
        if (!res.ok) { showStatus(data.error || 'failed', 'error'); resetBtn(btn); return; }

        const poll = setInterval(async () => {
            try {
                const s = await (await fetch('/api/training_status')).json();
                document.getElementById('progress-fill').style.width = s.progress + '%';
                document.getElementById('progress-text').textContent = s.progress + '%';
                document.getElementById('progress-message').textContent = s.message;
                if (s.complete) {
                    clearInterval(poll);
                    showStatus(`ensemble trained for ${currentTicker}!`, 'success');
                    document.getElementById('predict-section').classList.remove('hidden');
                    btn.querySelector('.btn-text').textContent = 'retrain ensemble';
                    btn.querySelector('.btn-loader').classList.add('hidden');
                    btn.disabled = false;
                    if (s.backtest) renderBacktest(s.backtest);
                }
                if (s.error) { clearInterval(poll); showStatus('error: ' + s.error, 'error'); resetBtn(btn); }
            } catch(e) {}
        }, 1000);
    } catch (err) { showStatus('network error: ' + err.message, 'error'); resetBtn(btn); }
}

function resetBtn(btn) {
    btn.querySelector('.btn-text').textContent = 'train ensemble';
    btn.querySelector('.btn-loader').classList.add('hidden');
    btn.disabled = false;
}

function renderBacktest(bt) {
    document.getElementById('backtest-section').classList.remove('hidden');

    const d = bt.ensemble_direction_accuracy;
    const badge = document.getElementById('backtest-confidence');
    let conf = d >= 65 ? 'high' : d >= 55 ? 'medium' : 'low';
    badge.className = 'confidence-badge confidence-' + conf;
    badge.textContent = conf + ' confidence';

    //individual model cards
    const cards = document.getElementById('model-cards');
    cards.innerHTML = (bt.individual_models || []).map((m, i) => {
        const col = m.direction_accuracy >= 60 ? 'var(--green)' : m.direction_accuracy >= 52 ? 'var(--amber)' : 'var(--red)';
        return `<div class="model-card">
            <div class="model-card-name" style="color:${MODEL_COLOURS[i]}">${m.name}</div>
            <div class="model-card-acc" style="color:${col}">${m.direction_accuracy}%</div>
            <div class="stat-label">direction accuracy</div>
        </div>`;
    }).join('');

    document.getElementById('bt-direction').textContent = d + '%';
    document.getElementById('bt-direction').style.color = d >= 65 ? 'var(--green)' : d >= 55 ? 'var(--amber)' : 'var(--red)';
    document.getElementById('bt-mae').textContent = '$' + bt.mae;
    document.getElementById('bt-mape').textContent = bt.mape + '%';
    document.getElementById('bt-rmse').textContent = '$' + bt.rmse;

    if (bt.val_predictions && bt.val_actual) {
        const ctx = document.getElementById('backtest-chart').getContext('2d');
        if (backtestChart) backtestChart.destroy();
        backtestChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: bt.val_actual.map((_, i) => i + 1),
                datasets: [
                    { label: 'actual', data: bt.val_actual, borderColor: '#4a7cff', borderWidth: 1.5, pointRadius: 0, tension: 0.1, fill: false },
                    { label: 'ensemble predicted', data: bt.val_predictions, borderColor: '#2dd4a0', borderWidth: 1.5, pointRadius: 0, tension: 0.1, borderDash: [4,3], fill: false }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { labels: { font: { family: 'JetBrains Mono', size: 10 }, color: '#8888a0', boxWidth: 12 } },
                    tooltip: { backgroundColor: '#16161f', borderColor: '#2a2a3a', borderWidth: 1, titleFont: { family: 'JetBrains Mono', size: 11 }, bodyFont: { family: 'JetBrains Mono', size: 12 }, titleColor: '#8888a0', bodyColor: '#e8e8f0', displayColors: false, callbacks: { label: c => c.dataset.label + ': $' + c.parsed.y.toFixed(2) } } },
                scales: {
                    x: { title: { display: true, text: 'validation days', color: '#55556a', font: { family: 'JetBrains Mono', size: 10 } }, grid: { color: 'rgba(42,42,58,0.3)' }, ticks: { font: { family: 'JetBrains Mono', size: 9 }, color: '#55556a', maxTicksLimit: 10 } },
                    y: { grid: { color: 'rgba(42,42,58,0.3)' }, ticks: { font: { family: 'JetBrains Mono', size: 10 }, color: '#55556a', callback: v => '$' + v.toFixed(0) } }
                }
            }
        });
    }
}

async function runPrediction(timeframe, btnEl) {
    if (!currentTicker) return;
    document.querySelectorAll('.timeframe-btn').forEach(b => { b.classList.remove('active'); b.classList.add('loading'); });
    btnEl.classList.add('active');
    showStatus(`running ${timeframe} ensemble prediction for ${currentTicker}...`);

    try {
        const res = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ticker: currentTicker, timeframe }) });
        const data = await res.json();
        document.querySelectorAll('.timeframe-btn').forEach(b => b.classList.remove('loading'));
        if (!res.ok) { showStatus(data.error || 'prediction failed', 'error'); return; }

        document.getElementById('prediction-results').classList.remove('hidden');
        renderVotePanel(data);
        renderPredictionChart(data);
        renderAnalysis(data);
        showStatus(`${timeframe} ensemble prediction complete`, 'success');
    } catch (err) {
        document.querySelectorAll('.timeframe-btn').forEach(b => b.classList.remove('loading'));
        showStatus('error: ' + err.message, 'error');
    }
}

function renderVotePanel(data) {
    const panel = document.getElementById('vote-panel');
    if (!data.model_votes || !data.model_votes.length) { panel.innerHTML = ''; return; }

    panel.innerHTML = data.model_votes.map((v, i) => {
        const dir = v.direction;
        const sign = v.change_pct >= 0 ? '+' : '';
        return `<div class="vote-card ${dir}">
            <div class="vote-model-name" style="color:${MODEL_COLOURS[i]}">${v.name}</div>
            <div class="vote-direction ${dir}">${dir === 'up' ? '&#9650; UP' : '&#9660; DOWN'}</div>
            <div class="vote-change" style="color:${dir === 'up' ? 'var(--green)' : 'var(--red)'}">${sign}${v.change_pct}%</div>
            <div class="vote-price">$${v.final_price}</div>
        </div>`;
    }).join('');
}

function renderPredictionChart(data) {
    const ctx = document.getElementById('prediction-chart').getContext('2d');
    if (predictionChart) predictionChart.destroy();

    const hc = Math.min(30, fullChartData.dates.length);
    const hd = fullChartData.dates.slice(-hc);
    const hp = fullChartData.close.slice(-hc);
    const pad = Array(hd.length - 1).fill(null);

    const allDates = [...hd, ...data.predictions.dates];
    const histDS = [...hp, ...Array(data.predictions.dates.length).fill(null)];
    const ensembleDS = [...pad, hp[hp.length-1], ...data.predictions.prices];
    const upperDS = [...pad, hp[hp.length-1], ...data.predictions.upper_band];
    const lowerDS = [...pad, hp[hp.length-1], ...data.predictions.lower_band];

    const datasets = [
        { label: 'historical', data: histDS, borderColor: '#4a7cff', borderWidth: 1.5, pointRadius: 0, tension: 0.1, fill: false },
        { label: '95% CI upper', data: upperDS, borderColor: 'rgba(45,212,160,0.2)', backgroundColor: 'rgba(45,212,160,0.05)', borderWidth: 1, borderDash: [2,4], pointRadius: 0, tension: 0.1, fill: '+1' },
        { label: 'ensemble', data: ensembleDS, borderColor: '#2dd4a0', borderWidth: 2.5, pointRadius: 0, tension: 0.1, fill: false },
        { label: '95% CI lower', data: lowerDS, borderColor: 'rgba(45,212,160,0.2)', borderWidth: 1, borderDash: [2,4], pointRadius: 0, tension: 0.1, fill: false },
    ];

    //add individual model lines
    if (data.model_votes) {
        data.model_votes.forEach((v, i) => {
            const modelDS = [...pad, hp[hp.length-1], ...v.prices];
            datasets.push({
                label: v.name,
                data: modelDS,
                borderColor: MODEL_COLOURS[i],
                borderWidth: 1,
                borderDash: [3, 3],
                pointRadius: 0,
                tension: 0.1,
                fill: false,
                hidden: false
            });
        });
    }

    predictionChart = new Chart(ctx, {
        type: 'line', data: { labels: allDates, datasets },
        options: {
            responsive: true, maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            plugins: {
                legend: { labels: { font: { family: 'JetBrains Mono', size: 10 }, color: '#8888a0', boxWidth: 12, padding: 10, filter: item => !item.text.includes('CI') } },
                tooltip: { backgroundColor: '#16161f', borderColor: '#2a2a3a', borderWidth: 1, titleFont: { family: 'JetBrains Mono', size: 11 }, bodyFont: { family: 'JetBrains Mono', size: 12 }, titleColor: '#8888a0', bodyColor: '#e8e8f0', displayColors: true, callbacks: { label: c => c.parsed.y != null ? c.dataset.label + ': $' + c.parsed.y.toFixed(2) : '' } }
            },
            scales: {
                x: { grid: { color: 'rgba(42,42,58,0.3)' }, ticks: { font: { family: 'JetBrains Mono', size: 10 }, color: '#55556a', maxTicksLimit: 8 } },
                y: { grid: { color: 'rgba(42,42,58,0.3)' }, ticks: { font: { family: 'JetBrains Mono', size: 10 }, color: '#55556a', callback: v => '$' + v.toFixed(0) } }
            }
        }
    });
}

function renderAnalysis(data) {
    const a = data.analysis;
    const card = document.getElementById('analysis-card');
    const labels = { buy_then_sell: 'buy then sell', sell_then_buy: 'sell then rebuy', buy: 'buy', sell: 'sell', hold: 'hold' };
    const confCol = a.model_confidence === 'high' ? 'var(--green)' : a.model_confidence === 'medium' ? 'var(--amber)' : 'var(--red)';
    const newsCol = a.news_bias === 'positive' ? 'var(--green)' : a.news_bias === 'negative' ? 'var(--red)' : 'var(--text-muted)';

    let saveBtn = '';
    if (a.action !== 'hold') {
        const td = { ticker: data.ticker, action: a.action, buy_date: a.buy_date||'-', buy_price: a.buy_price||'-', sell_date: a.sell_date||'-', sell_price: a.sell_price||'-', potential_gain: a.potential_profit_pct ? a.potential_profit_pct+'%' : a.overall_change_pct+'%', timeframe: data.timeframe };
        saveBtn = `<button class="save-trade-btn" onclick='saveTrade(${JSON.stringify(td)}, this)'>save to trade table</button>`;
    }

    card.innerHTML = `
        <span class="analysis-action action-${a.action}">${labels[a.action] || a.action}</span>
        <p class="analysis-reason">${a.reason}</p>
        <div class="analysis-stats">
            <div class="stat-item"><span class="stat-label">predicted low</span><span class="stat-value" style="color:var(--red)">$${a.predicted_low}</span></div>
            <div class="stat-item"><span class="stat-label">predicted high</span><span class="stat-value" style="color:var(--green)">$${a.predicted_high}</span></div>
            <div class="stat-item"><span class="stat-label">overall change</span><span class="stat-value" style="color:${a.overall_change_pct>=0?'var(--green)':'var(--red)'}">${a.overall_change_pct>=0?'+':''}${a.overall_change_pct}%</span></div>
            <div class="stat-item"><span class="stat-label">ensemble confidence</span><span class="stat-value" style="color:${confCol}">${a.model_confidence} (${a.direction_accuracy}%)</span></div>
            <div class="stat-item"><span class="stat-label">news sentiment</span><span class="stat-value" style="color:${newsCol}">${a.news_bias} (${a.news_score>=0?'+':''}${a.news_score})</span></div>
            <div class="stat-item"><span class="stat-label">model consensus</span><span class="stat-value">${a.consensus}</span></div>
        </div>
        ${saveBtn}
    `;
}

async function saveTrade(td, btn) {
    try {
        const r = await fetch('/api/save_trade', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(td) });
        if (r.ok) { btn.textContent = 'saved!'; btn.classList.add('saved'); loadTrades(); }
    } catch(e) { showStatus('failed to save', 'error'); }
}

async function loadTrades() {
    try {
        const trades = await (await fetch('/api/trades')).json();
        const table = document.getElementById('trades-table');
        const noMsg = document.getElementById('no-trades-msg');
        const tbody = document.getElementById('trades-body');
        if (!trades.length) { table.classList.add('hidden'); noMsg.classList.remove('hidden'); return; }
        table.classList.remove('hidden'); noMsg.classList.add('hidden');
        tbody.innerHTML = trades.map(t => {
            const c = (t.action==='buy'||t.action==='buy_then_sell') ? 'var(--green)' : (t.action==='sell'||t.action==='sell_then_buy') ? 'var(--red)' : 'var(--amber)';
            return `<tr><td style="color:var(--accent);font-weight:600">${t.ticker}</td><td style="color:${c};font-weight:600;text-transform:uppercase;font-size:11px">${t.action}</td><td>${t.buy_date}</td><td>${t.buy_price!=='-'?'$'+t.buy_price:'-'}</td><td>${t.sell_date}</td><td>${t.sell_price!=='-'?'$'+t.sell_price:'-'}</td><td style="color:var(--green)">${t.potential_gain}</td><td>${t.saved_at||'-'}</td></tr>`;
        }).join('');
    } catch(e) {}
}

async function clearTrades() {
    try { await fetch('/api/clear_trades', { method: 'POST' }); loadTrades(); showStatus('trades cleared', 'success'); }
    catch(e) { showStatus('failed', 'error'); }
}

loadTrades();
