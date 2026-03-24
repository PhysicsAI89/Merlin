/*
merlin 5.1 - frontend logic
handles fetching, charts, training, backtesting, news and predictions
*/

let currentTicker = '';
let priceChart = null;
let predictionChart = null;
let backtestChart = null;
let fullChartData = null;

//enter key triggers fetch
document.getElementById('ticker-input').addEventListener('keydown', function(e) {
    if (e.key === 'Enter') fetchData();
});

//chart range buttons
document.querySelectorAll('.chart-range-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        document.querySelectorAll('.chart-range-btn').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
        if (fullChartData) renderChart(this.dataset.range);
    });
});


function showStatus(msg, type = '') {
    const bar = document.getElementById('status-bar');
    const text = document.getElementById('status-text');
    bar.className = 'status-bar ' + type;
    text.textContent = msg;
    bar.classList.remove('hidden');
    if (type === 'success') {
        setTimeout(() => bar.classList.add('hidden'), 4000);
    }
}


async function fetchData() {
    const input = document.getElementById('ticker-input');
    const ticker = input.value.trim().toUpperCase();
    if (!ticker) {
        showStatus('enter a ticker symbol first', 'error');
        return;
    }

    const btn = document.getElementById('fetch-btn');
    btn.querySelector('.btn-text').textContent = 'fetching...';
    btn.querySelector('.btn-loader').classList.remove('hidden');
    btn.disabled = true;

    try {
        showStatus(`fetching data for ${ticker}...`);

        const res = await fetch('/api/fetch_data', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker })
        });

        const data = await res.json();

        if (!res.ok) {
            showStatus(data.error || 'failed to fetch data', 'error');
            return;
        }

        currentTicker = data.ticker;
        fullChartData = data.chart_data;

        //update info cards
        document.getElementById('info-ticker').textContent = data.ticker;
        document.getElementById('info-price').textContent = '$' + data.current_price.toFixed(2);

        const changeEl = document.getElementById('info-change');
        const sign = data.change >= 0 ? '+' : '';
        changeEl.textContent = `${sign}$${data.change.toFixed(2)} (${sign}${data.change_pct}%)`;
        changeEl.className = 'info-value ' + (data.change >= 0 ? 'positive' : 'negative');
        document.getElementById('info-points').textContent = data.data_points.toLocaleString();

        //show sections
        document.getElementById('stock-info').classList.remove('hidden');
        document.getElementById('chart-section').classList.remove('hidden');
        document.getElementById('train-section').classList.remove('hidden');

        //hide prediction/backtest until model trained
        document.getElementById('predict-section').classList.add('hidden');
        document.getElementById('backtest-section').classList.add('hidden');
        document.getElementById('prediction-results').classList.add('hidden');

        //render news
        if (data.news && data.news.articles && data.news.articles.length > 0) {
            renderNews(data.news);
            document.getElementById('news-section').classList.remove('hidden');
        } else {
            document.getElementById('news-section').classList.add('hidden');
        }

        //render chart
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
    const list = document.getElementById('news-list');

    //set badge
    let badgeClass = 'neutral';
    let badgeText = 'neutral';
    if (news.overall_score > 0.3) { badgeClass = 'positive'; badgeText = 'positive'; }
    else if (news.overall_score < -0.3) { badgeClass = 'negative'; badgeText = 'negative'; }
    badge.className = 'news-badge ' + badgeClass;
    badge.textContent = badgeText + ' (' + news.overall_score.toFixed(1) + ')';

    //render articles
    list.innerHTML = news.articles.map(a => `
        <div class="news-item">
            <div class="news-sentiment-dot ${a.sentiment}"></div>
            <div class="news-content">
                <div class="news-title">
                    ${a.link ? `<a href="${a.link}" target="_blank" rel="noopener">${a.title}</a>` : a.title}
                </div>
                <div class="news-meta">${a.publisher}${a.date ? ' · ' + a.date : ''}</div>
            </div>
        </div>
    `).join('');
}


function renderChart(range) {
    if (!fullChartData) return;

    let dates = fullChartData.dates;
    let prices = fullChartData.close;
    let sma20 = fullChartData.sma_20;
    let sma50 = fullChartData.sma_50;

    const total = dates.length;
    let start = 0;
    switch(range) {
        case '3m': start = Math.max(0, total - 66); break;
        case '6m': start = Math.max(0, total - 132); break;
        case '1y': start = Math.max(0, total - 252); break;
        case 'all': start = 0; break;
    }

    dates = dates.slice(start);
    prices = prices.slice(start);
    sma20 = sma20 ? sma20.slice(start) : null;
    sma50 = sma50 ? sma50.slice(start) : null;

    const ctx = document.getElementById('price-chart').getContext('2d');
    if (priceChart) priceChart.destroy();

    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, 'rgba(74, 124, 255, 0.25)');
    gradient.addColorStop(1, 'rgba(74, 124, 255, 0.0)');

    const datasets = [
        {
            label: currentTicker + ' close',
            data: prices,
            borderColor: '#4a7cff',
            backgroundColor: gradient,
            borderWidth: 1.5,
            fill: true,
            pointRadius: 0,
            pointHitRadius: 10,
            tension: 0.1
        }
    ];

    if (sma20) {
        datasets.push({
            label: 'SMA 20',
            data: sma20,
            borderColor: 'rgba(240, 160, 48, 0.6)',
            borderWidth: 1,
            pointRadius: 0,
            fill: false,
            tension: 0.1,
            borderDash: [4, 3]
        });
    }

    if (sma50) {
        datasets.push({
            label: 'SMA 50',
            data: sma50,
            borderColor: 'rgba(45, 212, 160, 0.5)',
            borderWidth: 1,
            pointRadius: 0,
            fill: false,
            tension: 0.1,
            borderDash: [6, 4]
        });
    }

    priceChart = new Chart(ctx, {
        type: 'line',
        data: { labels: dates, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            plugins: {
                legend: {
                    labels: {
                        font: { family: 'JetBrains Mono', size: 10 },
                        color: '#8888a0',
                        boxWidth: 12,
                        padding: 12
                    }
                },
                tooltip: {
                    backgroundColor: '#16161f',
                    borderColor: '#2a2a3a',
                    borderWidth: 1,
                    titleFont: { family: 'JetBrains Mono', size: 11 },
                    bodyFont: { family: 'JetBrains Mono', size: 12 },
                    titleColor: '#8888a0',
                    bodyColor: '#e8e8f0',
                    padding: 10,
                    displayColors: false,
                    callbacks: {
                        label: ctx => ctx.parsed.y != null ? ctx.dataset.label + ': $' + ctx.parsed.y.toFixed(2) : ''
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(42, 42, 58, 0.3)' },
                    ticks: { font: { family: 'JetBrains Mono', size: 10 }, color: '#55556a', maxTicksLimit: 8 }
                },
                y: {
                    grid: { color: 'rgba(42, 42, 58, 0.3)' },
                    ticks: { font: { family: 'JetBrains Mono', size: 10 }, color: '#55556a', callback: v => '$' + v.toFixed(0) }
                }
            }
        }
    });
}


async function trainModel() {
    if (!currentTicker) {
        showStatus('fetch data for a stock first', 'error');
        return;
    }

    const btn = document.getElementById('train-btn');
    const epochs = parseInt(document.getElementById('epochs-input').value) || 50;

    btn.querySelector('.btn-text').textContent = 'training...';
    btn.querySelector('.btn-loader').classList.remove('hidden');
    btn.disabled = true;

    document.getElementById('training-progress').classList.remove('hidden');
    document.getElementById('backtest-section').classList.add('hidden');

    try {
        const res = await fetch('/api/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker: currentTicker, epochs })
        });

        const data = await res.json();
        if (!res.ok) {
            showStatus(data.error || 'training failed', 'error');
            btn.querySelector('.btn-text').textContent = 'train model';
            btn.querySelector('.btn-loader').classList.add('hidden');
            btn.disabled = false;
            return;
        }

        //poll for progress
        const pollInterval = setInterval(async () => {
            try {
                const statusRes = await fetch('/api/training_status');
                const status = await statusRes.json();

                document.getElementById('progress-fill').style.width = status.progress + '%';
                document.getElementById('progress-text').textContent = status.progress + '%';
                document.getElementById('progress-message').textContent = status.message;

                if (status.complete) {
                    clearInterval(pollInterval);
                    showStatus(`model trained for ${currentTicker}!`, 'success');
                    document.getElementById('predict-section').classList.remove('hidden');
                    btn.querySelector('.btn-text').textContent = 'retrain model';
                    btn.querySelector('.btn-loader').classList.add('hidden');
                    btn.disabled = false;

                    //show backtest results
                    if (status.backtest) {
                        renderBacktest(status.backtest);
                    }
                }

                if (status.error) {
                    clearInterval(pollInterval);
                    showStatus('training error: ' + status.error, 'error');
                    btn.querySelector('.btn-text').textContent = 'train model';
                    btn.querySelector('.btn-loader').classList.add('hidden');
                    btn.disabled = false;
                }
            } catch (e) { /* keep polling */ }
        }, 1000);

    } catch (err) {
        showStatus('network error: ' + err.message, 'error');
        btn.querySelector('.btn-text').textContent = 'train model';
        btn.querySelector('.btn-loader').classList.add('hidden');
        btn.disabled = false;
    }
}


function renderBacktest(bt) {
    document.getElementById('backtest-section').classList.remove('hidden');

    //confidence badge
    const badge = document.getElementById('backtest-confidence');
    const dirAcc = bt.direction_accuracy;
    let conf = 'low';
    if (dirAcc >= 65) conf = 'high';
    else if (dirAcc >= 55) conf = 'medium';
    badge.className = 'confidence-badge confidence-' + conf;
    badge.textContent = conf + ' confidence';

    //stats
    document.getElementById('bt-direction').textContent = dirAcc + '%';
    document.getElementById('bt-direction').style.color =
        dirAcc >= 65 ? 'var(--green)' : dirAcc >= 55 ? 'var(--amber)' : 'var(--red)';
    document.getElementById('bt-mae').textContent = '$' + bt.mae;
    document.getElementById('bt-mape').textContent = bt.mape + '%';
    document.getElementById('bt-rmse').textContent = '$' + bt.rmse;

    //backtest chart (predicted vs actual on validation data)
    if (bt.val_predictions && bt.val_actual) {
        const ctx = document.getElementById('backtest-chart').getContext('2d');
        if (backtestChart) backtestChart.destroy();

        const labels = bt.val_actual.map((_, i) => i + 1);

        backtestChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    {
                        label: 'actual',
                        data: bt.val_actual,
                        borderColor: '#4a7cff',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        tension: 0.1,
                        fill: false
                    },
                    {
                        label: 'predicted',
                        data: bt.val_predictions,
                        borderColor: '#2dd4a0',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        tension: 0.1,
                        borderDash: [4, 3],
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            font: { family: 'JetBrains Mono', size: 10 },
                            color: '#8888a0',
                            boxWidth: 12,
                            padding: 12
                        }
                    },
                    tooltip: {
                        backgroundColor: '#16161f',
                        borderColor: '#2a2a3a',
                        borderWidth: 1,
                        titleFont: { family: 'JetBrains Mono', size: 11 },
                        bodyFont: { family: 'JetBrains Mono', size: 12 },
                        titleColor: '#8888a0',
                        bodyColor: '#e8e8f0',
                        displayColors: false,
                        callbacks: {
                            label: ctx => ctx.dataset.label + ': $' + ctx.parsed.y.toFixed(2)
                        }
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: { display: true, text: 'validation days', color: '#55556a', font: { family: 'JetBrains Mono', size: 10 } },
                        grid: { color: 'rgba(42, 42, 58, 0.3)' },
                        ticks: { font: { family: 'JetBrains Mono', size: 9 }, color: '#55556a', maxTicksLimit: 10 }
                    },
                    y: {
                        grid: { color: 'rgba(42, 42, 58, 0.3)' },
                        ticks: { font: { family: 'JetBrains Mono', size: 10 }, color: '#55556a', callback: v => '$' + v.toFixed(0) }
                    }
                }
            }
        });
    }
}


async function runPrediction(timeframe, btnEl) {
    if (!currentTicker) return;

    document.querySelectorAll('.timeframe-btn').forEach(b => {
        b.classList.remove('active');
        b.classList.add('loading');
    });
    btnEl.classList.add('active');

    showStatus(`running ${timeframe} prediction for ${currentTicker}...`);

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker: currentTicker, timeframe })
        });

        const data = await res.json();
        document.querySelectorAll('.timeframe-btn').forEach(b => b.classList.remove('loading'));

        if (!res.ok) {
            showStatus(data.error || 'prediction failed', 'error');
            return;
        }

        document.getElementById('prediction-results').classList.remove('hidden');
        renderPredictionChart(data);
        renderAnalysis(data);
        showStatus(`${timeframe} prediction complete for ${currentTicker}`, 'success');

    } catch (err) {
        document.querySelectorAll('.timeframe-btn').forEach(b => b.classList.remove('loading'));
        showStatus('prediction error: ' + err.message, 'error');
    }
}


function renderPredictionChart(data) {
    const ctx = document.getElementById('prediction-chart').getContext('2d');
    if (predictionChart) predictionChart.destroy();

    const histCount = Math.min(30, fullChartData.dates.length);
    const histDates = fullChartData.dates.slice(-histCount);
    const histPrices = fullChartData.close.slice(-histCount);

    const allDates = [...histDates, ...data.predictions.dates];
    const histDataset = [...histPrices, ...Array(data.predictions.dates.length).fill(null)];
    const predDataset = [...Array(histDates.length - 1).fill(null), histPrices[histPrices.length - 1], ...data.predictions.prices];

    const gradient = ctx.createLinearGradient(0, 0, 0, 250);
    gradient.addColorStop(0, 'rgba(45, 212, 160, 0.15)');
    gradient.addColorStop(1, 'rgba(45, 212, 160, 0.0)');

    predictionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: allDates,
            datasets: [
                {
                    label: 'historical',
                    data: histDataset,
                    borderColor: '#4a7cff',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.1,
                    fill: false
                },
                {
                    label: 'predicted',
                    data: predDataset,
                    borderColor: '#2dd4a0',
                    backgroundColor: gradient,
                    borderWidth: 2,
                    borderDash: [4, 3],
                    pointRadius: 0,
                    tension: 0.1,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            plugins: {
                legend: {
                    labels: {
                        font: { family: 'JetBrains Mono', size: 10 },
                        color: '#8888a0',
                        boxWidth: 12,
                        padding: 16
                    }
                },
                tooltip: {
                    backgroundColor: '#16161f',
                    borderColor: '#2a2a3a',
                    borderWidth: 1,
                    titleFont: { family: 'JetBrains Mono', size: 11 },
                    bodyFont: { family: 'JetBrains Mono', size: 12 },
                    titleColor: '#8888a0',
                    bodyColor: '#e8e8f0',
                    displayColors: false,
                    callbacks: {
                        label: ctx => ctx.parsed.y != null ? ctx.dataset.label + ': $' + ctx.parsed.y.toFixed(2) : ''
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(42, 42, 58, 0.3)' },
                    ticks: { font: { family: 'JetBrains Mono', size: 10 }, color: '#55556a', maxTicksLimit: 8 }
                },
                y: {
                    grid: { color: 'rgba(42, 42, 58, 0.3)' },
                    ticks: { font: { family: 'JetBrains Mono', size: 10 }, color: '#55556a', callback: v => '$' + v.toFixed(0) }
                }
            }
        }
    });
}


function renderAnalysis(data) {
    const analysis = data.analysis;
    const card = document.getElementById('analysis-card');

    const actionLabels = {
        'buy_then_sell': 'buy then sell',
        'sell_then_buy': 'sell then rebuy',
        'buy': 'buy',
        'sell': 'sell',
        'hold': 'hold'
    };

    const confColour = analysis.model_confidence === 'high' ? 'var(--green)' :
                       analysis.model_confidence === 'medium' ? 'var(--amber)' : 'var(--red)';

    const newsColour = analysis.news_bias === 'positive' ? 'var(--green)' :
                       analysis.news_bias === 'negative' ? 'var(--red)' : 'var(--text-muted)';

    let statsHtml = `
        <div class="analysis-stats">
            <div class="stat-item">
                <span class="stat-label">predicted low</span>
                <span class="stat-value" style="color: var(--red)">$${analysis.predicted_low}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">predicted high</span>
                <span class="stat-value" style="color: var(--green)">$${analysis.predicted_high}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">overall change</span>
                <span class="stat-value" style="color: ${analysis.overall_change_pct >= 0 ? 'var(--green)' : 'var(--red)'}">
                    ${analysis.overall_change_pct >= 0 ? '+' : ''}${analysis.overall_change_pct}%
                </span>
            </div>
            <div class="stat-item">
                <span class="stat-label">model confidence</span>
                <span class="stat-value" style="color: ${confColour}">${analysis.model_confidence} (${analysis.direction_accuracy}%)</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">news sentiment</span>
                <span class="stat-value" style="color: ${newsColour}">${analysis.news_bias} (${analysis.news_score >= 0 ? '+' : ''}${analysis.news_score})</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">end price</span>
                <span class="stat-value">$${analysis.final_predicted_price}</span>
            </div>
        </div>
    `;

    let saveBtn = '';
    if (analysis.action !== 'hold') {
        const tradeData = {
            ticker: data.ticker,
            action: analysis.action,
            buy_date: analysis.buy_date || '-',
            buy_price: analysis.buy_price || '-',
            sell_date: analysis.sell_date || '-',
            sell_price: analysis.sell_price || '-',
            potential_gain: analysis.potential_profit_pct ? analysis.potential_profit_pct + '%' : analysis.overall_change_pct + '%',
            timeframe: data.timeframe
        };
        saveBtn = `<button class="save-trade-btn" onclick='saveTrade(${JSON.stringify(tradeData)}, this)'>save to trade table</button>`;
    }

    card.innerHTML = `
        <span class="analysis-action action-${analysis.action}">${actionLabels[analysis.action] || analysis.action}</span>
        <p class="analysis-reason">${analysis.reason}</p>
        ${statsHtml}
        ${saveBtn}
    `;
}


async function saveTrade(tradeData, btn) {
    try {
        const res = await fetch('/api/save_trade', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(tradeData)
        });
        if (res.ok) {
            btn.textContent = 'saved!';
            btn.classList.add('saved');
            loadTrades();
        }
    } catch (err) {
        showStatus('failed to save trade', 'error');
    }
}


async function loadTrades() {
    try {
        const res = await fetch('/api/trades');
        const trades = await res.json();

        const table = document.getElementById('trades-table');
        const noMsg = document.getElementById('no-trades-msg');
        const tbody = document.getElementById('trades-body');

        if (trades.length === 0) {
            table.classList.add('hidden');
            noMsg.classList.remove('hidden');
            return;
        }

        table.classList.remove('hidden');
        noMsg.classList.add('hidden');

        tbody.innerHTML = trades.map(t => {
            const actionColour = (t.action === 'buy' || t.action === 'buy_then_sell') ? 'var(--green)' :
                                 (t.action === 'sell' || t.action === 'sell_then_buy') ? 'var(--red)' : 'var(--amber)';
            return `<tr>
                <td style="color: var(--accent); font-weight: 600">${t.ticker}</td>
                <td style="color: ${actionColour}; font-weight: 600; text-transform: uppercase; font-size: 11px">${t.action}</td>
                <td>${t.buy_date}</td>
                <td>${t.buy_price !== '-' ? '$' + t.buy_price : '-'}</td>
                <td>${t.sell_date}</td>
                <td>${t.sell_price !== '-' ? '$' + t.sell_price : '-'}</td>
                <td style="color: var(--green)">${t.potential_gain}</td>
                <td>${t.saved_at || '-'}</td>
            </tr>`;
        }).join('');
    } catch (err) { /* silent */ }
}


async function clearTrades() {
    try {
        await fetch('/api/clear_trades', { method: 'POST' });
        loadTrades();
        showStatus('trades cleared', 'success');
    } catch (err) {
        showStatus('failed to clear trades', 'error');
    }
}

//load trades on page load
loadTrades();
