/*
merlin 6.0 - ensemble + screener + insider trading tab
*/
let currentTicker='',priceChart=null,predictionChart=null,backtestChart=null,fullChartData=null;
const MC=['#4a7cff','#f0a030','#a07cff'];

document.getElementById('ticker-input').addEventListener('keydown',e=>{if(e.key==='Enter')fetchData()});
document.querySelectorAll('.chart-range-btn').forEach(b=>{b.addEventListener('click',function(){document.querySelectorAll('.chart-range-btn').forEach(x=>x.classList.remove('active'));this.classList.add('active');if(fullChartData)renderChart(this.dataset.range)})});

function switchTab(tab,el){document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));document.querySelectorAll('.tab-content').forEach(c=>c.classList.add('hidden'));document.getElementById('tab-'+tab).classList.remove('hidden');el.classList.add('active')}
function showStatus(msg,type=''){const b=document.getElementById('status-bar');b.className='status-bar '+type;document.getElementById('status-text').textContent=msg;b.classList.remove('hidden');if(type==='success')setTimeout(()=>b.classList.add('hidden'),4000)}
function fmtVal(v){if(!v||v===0)return'$0';if(Math.abs(v)>=1e6)return'$'+(v/1e6).toFixed(1)+'M';if(Math.abs(v)>=1e3)return'$'+(v/1e3).toFixed(0)+'K';return'$'+v.toFixed(0)}

async function fetchData(){
    const ticker=document.getElementById('ticker-input').value.trim().toUpperCase();
    if(!ticker){showStatus('enter a ticker first','error');return}
    const btn=document.getElementById('fetch-btn');
    btn.querySelector('.btn-text').textContent='fetching...';btn.querySelector('.btn-loader').classList.remove('hidden');btn.disabled=true;
    try{
        showStatus(`fetching ${ticker}...`);
        const data=await(await fetch('/api/fetch_data',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ticker})})).json();
        if(data.error){showStatus(data.error,'error');return}
        currentTicker=data.ticker;fullChartData=data.chart_data;
        document.getElementById('info-ticker').textContent=data.ticker;
        document.getElementById('info-price').textContent='$'+data.current_price.toFixed(2);
        const ce=document.getElementById('info-change'),s=data.change>=0?'+':'';
        ce.textContent=`${s}$${data.change.toFixed(2)} (${s}${data.change_pct}%)`;
        ce.className='info-value '+(data.change>=0?'positive':'negative');
        document.getElementById('info-points').textContent=data.data_points.toLocaleString();
        ['stock-info','chart-section','train-section'].forEach(id=>document.getElementById(id).classList.remove('hidden'));
        ['predict-section','backtest-section','prediction-results'].forEach(id=>document.getElementById(id).classList.add('hidden'));
        if(data.fundamentals)renderFundamentals(data.fundamentals);
        if(data.insider)renderInsider(data.insider);
        if(data.news&&data.news.articles&&data.news.articles.length>0){renderNews(data.news);document.getElementById('news-section').classList.remove('hidden')}else document.getElementById('news-section').classList.add('hidden');
        document.querySelector('.chart-range-btn.active').click();
        showStatus(`loaded ${data.data_points} points for ${data.ticker}`,'success');
    }catch(e){showStatus('error: '+e.message,'error')}
    finally{btn.querySelector('.btn-text').textContent='fetch data';btn.querySelector('.btn-loader').classList.add('hidden');btn.disabled=false}
}

function renderFundamentals(f){
    const sec=document.getElementById('fundamentals-section');if(!f||f.error){sec.classList.add('hidden');return}sec.classList.remove('hidden');
    const a=f.assessment||{};
    const vb=document.getElementById('fund-verdict');vb.className='fund-verdict-badge '+(a.verdict||'unknown');
    const vl={undervalued:'potentially undervalued',overvalued:'potentially overvalued',fair_value:'fairly valued',unknown:'unknown'};
    vb.textContent=vl[a.verdict]||'unknown';
    const sc=a.strength==='strong'?'var(--green)':a.strength==='moderate'?'var(--amber)':'var(--red)';
    document.getElementById('fund-assessment').innerHTML=`<div class="fund-assessment-summary">${a.summary||''}</div><div class="fund-strength"><span class="label">fundamental strength: </span><span style="color:${sc};font-weight:600">${a.strength||'?'}</span> — ${a.strength_text||''}</div>`;
    const pc=v=>!v&&v!==0?'var(--text-secondary)':v;
    const peC=f.pe_ratio?(f.pe_ratio<0?'var(--red)':f.pe_ratio<20?'var(--green)':f.pe_ratio<40?'var(--amber)':'var(--red)'):'var(--text-secondary)';
    const betaC=f.beta?(f.beta>1.5?'var(--red)':f.beta>1.1?'var(--amber)':'var(--green)'):'var(--text-secondary)';
    document.getElementById('fund-grid').innerHTML=`
        <div class="fund-card"><span class="fund-card-label">market cap</span><span class="fund-card-value">${f.market_cap_str||'n/a'}</span><div class="fund-card-sub">${f.market_cap_class||''}</div></div>
        <div class="fund-card"><span class="fund-card-label">P/E ratio</span><span class="fund-card-value" style="color:${peC}">${f.pe_ratio||'n/a'}</span><div class="fund-card-sub">${f.forward_pe?'fwd: '+f.forward_pe:''}</div></div>
        <div class="fund-card"><span class="fund-card-label">beta</span><span class="fund-card-value" style="color:${betaC}">${f.beta||'n/a'}</span><div class="fund-card-sub">volatility vs market</div></div>
        <div class="fund-card"><span class="fund-card-label">dividend yield</span><span class="fund-card-value">${f.dividend_yield?f.dividend_yield+'%':'0%'}</span></div>
        <div class="fund-card"><span class="fund-card-label">avg volume</span><span class="fund-card-value">${f.avg_volume_str||'n/a'}</span></div>
        <div class="fund-card"><span class="fund-card-label">52w range</span><span class="fund-card-value">${f.range_position||0}%</span><div class="fund-card-sub">$${f.fifty_two_low||'?'} — $${f.fifty_two_high||'?'}</div></div>
        <div class="fund-card"><span class="fund-card-label">profit margin</span><span class="fund-card-value" style="color:${f.profit_margins&&f.profit_margins>15?'var(--green)':f.profit_margins&&f.profit_margins>0?'var(--amber)':'var(--red)'}">${f.profit_margins!=null?f.profit_margins+'%':'n/a'}</span></div>
        <div class="fund-card"><span class="fund-card-label">debt/equity</span><span class="fund-card-value" style="color:${f.debt_to_equity!=null?(f.debt_to_equity<80?'var(--green)':f.debt_to_equity<150?'var(--amber)':'var(--red)'):'var(--text-secondary)'}">${f.debt_to_equity||'n/a'}</span></div>`;
    document.getElementById('fund-points').innerHTML=(a.points||[]).map(p=>`<div class="fund-point"><div class="fund-point-dot ${p.type}"></div><span>${p.text}</span></div>`).join('');
}

function renderInsider(ins){
    const sec=document.getElementById('insider-section');if(!ins.transactions||!ins.transactions.length){sec.classList.add('hidden');return}sec.classList.remove('hidden');
    const badge=document.getElementById('insider-badge'),s=ins.sentiment;
    badge.className='news-badge '+(s==='bullish'?'positive':s==='bearish'?'negative':'neutral');badge.textContent=s;
    document.getElementById('insider-summary').innerHTML=`<div class="stat-item"><span class="stat-label">exec buys</span><span class="stat-value" style="color:var(--green)">${ins.exec_buys}</span></div><div class="stat-item"><span class="stat-label">exec sells</span><span class="stat-value" style="color:var(--red)">${ins.exec_sells}</span></div><div class="stat-item"><span class="stat-label">total</span><span class="stat-value">${ins.all_buys+ins.all_sells}</span></div>`;
    document.getElementById('insider-list').innerHTML=ins.transactions.filter(t=>t.action!=='other').slice(0,15).map(t=>`<div class="insider-item"><span class="insider-name">${t.name}</span><span class="insider-title">${t.title}</span><span class="insider-action ${t.action}">${t.action}</span><span class="insider-details">${t.shares?t.shares.toLocaleString()+' shares':''}${t.value?' · '+fmtVal(t.value):''}</span></div>`).join('');
}

function renderNews(news){
    const badge=document.getElementById('news-badge');let bc=news.overall_score>0.3?'positive':news.overall_score<-0.3?'negative':'neutral';
    badge.className='news-badge '+bc;badge.textContent=bc+' ('+(news.overall_score>=0?'+':'')+news.overall_score.toFixed(1)+')';
    document.getElementById('news-list').innerHTML=news.articles.map(a=>`<div class="news-item"><div class="news-sentiment-dot ${a.sentiment}"></div><div class="news-content"><div class="news-title">${a.link?`<a href="${a.link}" target="_blank">${a.title}</a>`:a.title}</div><div class="news-meta">${a.publisher}${a.date?' &middot; '+a.date:''}</div></div></div>`).join('');
}

function renderChart(range){
    if(!fullChartData)return;let d=fullChartData.dates,p=fullChartData.close,s20=fullChartData.sma_20,s50=fullChartData.sma_50;
    const t=d.length;let st=0;switch(range){case'3m':st=Math.max(0,t-66);break;case'6m':st=Math.max(0,t-132);break;case'1y':st=Math.max(0,t-252);break}
    d=d.slice(st);p=p.slice(st);s20=s20?s20.slice(st):null;s50=s50?s50.slice(st):null;
    const ctx=document.getElementById('price-chart').getContext('2d');if(priceChart)priceChart.destroy();
    const gr=ctx.createLinearGradient(0,0,0,300);gr.addColorStop(0,'rgba(74,124,255,0.25)');gr.addColorStop(1,'rgba(74,124,255,0)');
    const ds=[{label:currentTicker,data:p,borderColor:'#4a7cff',backgroundColor:gr,borderWidth:1.5,fill:true,pointRadius:0,tension:0.1}];
    if(s20)ds.push({label:'SMA 20',data:s20,borderColor:'rgba(240,160,48,0.6)',borderWidth:1,pointRadius:0,fill:false,tension:0.1,borderDash:[4,3]});
    if(s50)ds.push({label:'SMA 50',data:s50,borderColor:'rgba(45,212,160,0.5)',borderWidth:1,pointRadius:0,fill:false,tension:0.1,borderDash:[6,4]});
    priceChart=new Chart(ctx,{type:'line',data:{labels:d,datasets:ds},options:co('$')});
}
function co(pfx){return{responsive:true,maintainAspectRatio:false,interaction:{intersect:false,mode:'index'},plugins:{legend:{labels:{font:{family:'JetBrains Mono',size:10},color:'#8888a0',boxWidth:12}},tooltip:{backgroundColor:'#16161f',borderColor:'#2a2a3a',borderWidth:1,titleFont:{family:'JetBrains Mono',size:11},bodyFont:{family:'JetBrains Mono',size:12},titleColor:'#8888a0',bodyColor:'#e8e8f0',displayColors:false,callbacks:{label:c=>c.parsed.y!=null?c.dataset.label+': '+pfx+c.parsed.y.toFixed(2):''}}},scales:{x:{grid:{color:'rgba(42,42,58,0.3)'},ticks:{font:{family:'JetBrains Mono',size:10},color:'#55556a',maxTicksLimit:8}},y:{grid:{color:'rgba(42,42,58,0.3)'},ticks:{font:{family:'JetBrains Mono',size:10},color:'#55556a',callback:v=>pfx+v.toFixed(0)}}}}}

async function trainModel(){
    if(!currentTicker){showStatus('fetch data first','error');return}
    const btn=document.getElementById('train-btn'),ep=parseInt(document.getElementById('epochs-input').value)||50;
    btn.querySelector('.btn-text').textContent='training...';btn.querySelector('.btn-loader').classList.remove('hidden');btn.disabled=true;
    document.getElementById('training-progress').classList.remove('hidden');document.getElementById('backtest-section').classList.add('hidden');
    try{const r=await fetch('/api/train',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ticker:currentTicker,epochs:ep})});
    const d=await r.json();if(d.error){showStatus(d.error,'error');rb(btn);return}
    const poll=setInterval(async()=>{try{const s=await(await fetch('/api/training_status')).json();
    document.getElementById('progress-fill').style.width=s.progress+'%';document.getElementById('progress-text').textContent=s.progress+'%';document.getElementById('progress-message').textContent=s.message;
    if(s.complete){clearInterval(poll);showStatus('ensemble trained!','success');document.getElementById('predict-section').classList.remove('hidden');btn.querySelector('.btn-text').textContent='retrain';btn.querySelector('.btn-loader').classList.add('hidden');btn.disabled=false;if(s.backtest)renderBT(s.backtest)}
    if(s.error){clearInterval(poll);showStatus('error: '+s.error,'error');rb(btn)}}catch(e){}},1000)}catch(e){showStatus('error: '+e.message,'error');rb(btn)}
}
function rb(btn){btn.querySelector('.btn-text').textContent='train ensemble';btn.querySelector('.btn-loader').classList.add('hidden');btn.disabled=false}

function renderBT(bt){document.getElementById('backtest-section').classList.remove('hidden');const d=bt.ensemble_direction_accuracy,badge=document.getElementById('backtest-confidence');let c=d>=65?'high':d>=55?'medium':'low';badge.className='confidence-badge confidence-'+c;badge.textContent=c+' confidence';
document.getElementById('model-cards').innerHTML=(bt.individual_models||[]).map((m,i)=>{const cl=m.direction_accuracy>=60?'var(--green)':m.direction_accuracy>=52?'var(--amber)':'var(--red)';return`<div class="model-card"><div class="model-card-name" style="color:${MC[i]}">${m.name}</div><div class="model-card-acc" style="color:${cl}">${m.direction_accuracy}%</div><div class="stat-label">direction accuracy</div></div>`}).join('');
document.getElementById('bt-direction').textContent=d+'%';document.getElementById('bt-direction').style.color=d>=65?'var(--green)':d>=55?'var(--amber)':'var(--red)';
document.getElementById('bt-mae').textContent='$'+bt.mae;document.getElementById('bt-mape').textContent=bt.mape+'%';document.getElementById('bt-rmse').textContent='$'+bt.rmse;
if(bt.val_predictions&&bt.val_actual){const ctx=document.getElementById('backtest-chart').getContext('2d');if(backtestChart)backtestChart.destroy();
backtestChart=new Chart(ctx,{type:'line',data:{labels:bt.val_actual.map((_,i)=>i+1),datasets:[{label:'actual',data:bt.val_actual,borderColor:'#4a7cff',borderWidth:1.5,pointRadius:0,tension:0.1,fill:false},{label:'predicted',data:bt.val_predictions,borderColor:'#2dd4a0',borderWidth:1.5,pointRadius:0,tension:0.1,borderDash:[4,3],fill:false}]},options:co('$')})}}

async function runPrediction(tf,el){if(!currentTicker)return;document.querySelectorAll('.timeframe-btn').forEach(b=>{b.classList.remove('active');b.classList.add('loading')});el.classList.add('active');showStatus(`predicting ${tf}...`);
try{const data=await(await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ticker:currentTicker,timeframe:tf})})).json();
document.querySelectorAll('.timeframe-btn').forEach(b=>b.classList.remove('loading'));if(data.error){showStatus(data.error,'error');return}
document.getElementById('prediction-results').classList.remove('hidden');renderVP(data);renderPC(data);renderA(data);showStatus(tf+' done','success')}
catch(e){document.querySelectorAll('.timeframe-btn').forEach(b=>b.classList.remove('loading'));showStatus('error: '+e.message,'error')}}

function renderVP(data){const p=document.getElementById('vote-panel');if(!data.model_votes||!data.model_votes.length){p.innerHTML='';return}
p.innerHTML=data.model_votes.map((v,i)=>`<div class="vote-card ${v.direction}"><div class="vote-model-name" style="color:${MC[i]}">${v.name}</div><div class="vote-direction ${v.direction}">${v.direction==='up'?'&#9650; UP':'&#9660; DOWN'}</div><div class="vote-change" style="color:${v.direction==='up'?'var(--green)':'var(--red)'}">${v.change_pct>=0?'+':''}${v.change_pct}%</div><div class="vote-price">$${v.final_price}</div></div>`).join('')}

function renderPC(data){const ctx=document.getElementById('prediction-chart').getContext('2d');if(predictionChart)predictionChart.destroy();
const hc=Math.min(30,fullChartData.dates.length),hd=fullChartData.dates.slice(-hc),hp=fullChartData.close.slice(-hc),pad=Array(hd.length-1).fill(null),all=[...hd,...data.predictions.dates];
const ds=[{label:'historical',data:[...hp,...Array(data.predictions.dates.length).fill(null)],borderColor:'#4a7cff',borderWidth:1.5,pointRadius:0,tension:0.1,fill:false},
{label:'CI',data:[...pad,hp[hp.length-1],...data.predictions.upper_band],borderColor:'rgba(45,212,160,0.2)',backgroundColor:'rgba(45,212,160,0.05)',borderWidth:1,borderDash:[2,4],pointRadius:0,tension:0.1,fill:'+1'},
{label:'ensemble',data:[...pad,hp[hp.length-1],...data.predictions.prices],borderColor:'#2dd4a0',borderWidth:2.5,pointRadius:0,tension:0.1,fill:false},
{label:'CI_l',data:[...pad,hp[hp.length-1],...data.predictions.lower_band],borderColor:'rgba(45,212,160,0.2)',borderWidth:1,borderDash:[2,4],pointRadius:0,tension:0.1,fill:false}];
if(data.model_votes)data.model_votes.forEach((v,i)=>{ds.push({label:v.name,data:[...pad,hp[hp.length-1],...v.prices],borderColor:MC[i],borderWidth:1,borderDash:[3,3],pointRadius:0,tension:0.1,fill:false})});
predictionChart=new Chart(ctx,{type:'line',data:{labels:all,datasets:ds},options:{...co('$'),plugins:{...co('$').plugins,legend:{labels:{font:{family:'JetBrains Mono',size:10},color:'#8888a0',boxWidth:12,padding:10,filter:i=>!i.text.includes('CI')}}}}})}

function renderA(data){const a=data.analysis,card=document.getElementById('analysis-card');
const labels={buy_then_sell:'buy then sell',sell_then_buy:'sell then rebuy',buy:'buy',sell:'sell',hold:'hold'};
const cc=a.model_confidence==='high'?'var(--green)':a.model_confidence==='medium'?'var(--amber)':'var(--red)';
const nc=a.news_bias==='positive'?'var(--green)':a.news_bias==='negative'?'var(--red)':'var(--text-muted)';
const ic=a.insider_sentiment==='bullish'?'var(--green)':a.insider_sentiment==='bearish'?'var(--red)':'var(--text-muted)';
let sb='';if(a.action!=='hold'){const td={ticker:data.ticker,action:a.action,buy_date:a.buy_date||'-',buy_price:a.buy_price||'-',sell_date:a.sell_date||'-',sell_price:a.sell_price||'-',potential_gain:a.potential_profit_pct?a.potential_profit_pct+'%':a.overall_change_pct+'%',timeframe:data.timeframe};sb=`<button class="save-trade-btn" onclick='saveTrade(${JSON.stringify(td)},this)'>save to trade table</button>`}
card.innerHTML=`<span class="analysis-action action-${a.action}">${labels[a.action]||a.action}</span><p class="analysis-reason">${a.reason}</p>
<div class="analysis-stats"><div class="stat-item"><span class="stat-label">predicted low</span><span class="stat-value" style="color:var(--red)">$${a.predicted_low}</span></div>
<div class="stat-item"><span class="stat-label">predicted high</span><span class="stat-value" style="color:var(--green)">$${a.predicted_high}</span></div>
<div class="stat-item"><span class="stat-label">change</span><span class="stat-value" style="color:${a.overall_change_pct>=0?'var(--green)':'var(--red)'}">${a.overall_change_pct>=0?'+':''}${a.overall_change_pct}%</span></div>
<div class="stat-item"><span class="stat-label">confidence</span><span class="stat-value" style="color:${cc}">${a.model_confidence} (${a.direction_accuracy}%)</span></div>
<div class="stat-item"><span class="stat-label">news</span><span class="stat-value" style="color:${nc}">${a.news_bias}</span></div>
<div class="stat-item"><span class="stat-label">insider</span><span class="stat-value" style="color:${ic}">${a.insider_sentiment||'n/a'} (${a.exec_buys||0}B/${a.exec_sells||0}S)</span></div></div>${sb}`}

/* ===== SCREENER ===== */
async function runScreener(){const btn=document.getElementById('screener-btn'),tn=parseInt(document.getElementById('screener-topn').value),sc=parseInt(document.getElementById('screener-count').value);
btn.querySelector('.btn-text').textContent='scanning...';btn.querySelector('.btn-loader').classList.remove('hidden');btn.disabled=true;
document.getElementById('screener-progress').classList.remove('hidden');document.getElementById('screener-results').classList.add('hidden');
try{await fetch('/api/screener',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({top_n:tn,stock_count:sc})});
const poll=setInterval(async()=>{try{const s=await(await fetch('/api/screener_status')).json();
document.getElementById('screener-progress-fill').style.width=s.progress+'%';document.getElementById('screener-progress-text').textContent=s.progress+'%';document.getElementById('screener-progress-message').textContent=s.message;
if(s.complete){clearInterval(poll);btn.querySelector('.btn-text').textContent='run screener';btn.querySelector('.btn-loader').classList.add('hidden');btn.disabled=false;renderSR(s.results,s.total_scanned)}
if(s.error){clearInterval(poll);btn.querySelector('.btn-text').textContent='run screener';btn.querySelector('.btn-loader').classList.add('hidden');btn.disabled=false;showStatus('error: '+s.error,'error')}}catch(e){}},1500)}catch(e){btn.querySelector('.btn-text').textContent='run screener';btn.querySelector('.btn-loader').classList.add('hidden');btn.disabled=false}}

function renderSR(results,total){document.getElementById('screener-results').classList.remove('hidden');document.getElementById('screener-results-title').textContent=`top ${results.length} picks (${total} scanned)`;
const c=document.getElementById('screener-table-container');if(!results.length){c.innerHTML='<p class="no-trades">no strong signals found</p>';return}
let h=`<table class="screener-table"><thead><tr><th>#</th><th>ticker</th><th>price</th><th>1d</th><th>5d</th><th>RSI</th><th>P/E</th><th>beta</th><th>mkt cap</th><th>score /13</th><th>valuation</th><th>strength</th><th>signals</th></tr></thead><tbody>`;
results.forEach((r,i)=>{const dc=r.direction==='bullish'?'var(--green)':r.direction==='bearish'?'var(--red)':'var(--amber)';const vc=r.valuation==='undervalued'?'var(--green)':r.valuation==='overvalued'?'var(--red)':'var(--amber)';const sc=r.strength==='strong'?'var(--green)':r.strength==='weak'?'var(--red)':'var(--amber)';const sigs=(r.signals||[]).map(s=>`<span class="signal-tag">${s}</span>`).join('');
h+=`<tr><td>${i+1}</td><td><span class="ticker-link" onclick="loadFromScreener('${r.ticker}')">${r.ticker}</span></td><td>$${r.price}</td><td style="color:${r.change_1d>=0?'var(--green)':'var(--red)'}">${r.change_1d>=0?'+':''}${r.change_1d}%</td><td style="color:${r.change_5d>=0?'var(--green)':'var(--red)'}">${r.change_5d>=0?'+':''}${r.change_5d}%</td><td>${r.rsi}</td><td>${r.pe_ratio||'?'}</td><td>${r.beta||'?'}</td><td>${r.market_cap_str||'?'}</td><td style="color:${dc};font-weight:700">${r.combined_score||r.score}</td><td style="color:${vc};font-weight:600;text-transform:uppercase;font-size:10px">${r.valuation||'?'}</td><td style="color:${sc};font-weight:600;text-transform:uppercase;font-size:10px">${r.strength||'?'}</td><td><div class="screener-signals">${sigs}</div></td></tr>`});
h+='</tbody></table>';c.innerHTML=h}

function loadFromScreener(ticker){document.getElementById('ticker-input').value=ticker;
document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));document.querySelectorAll('.tab-content').forEach(c=>c.classList.add('hidden'));
document.getElementById('tab-analyse').classList.remove('hidden');document.querySelectorAll('.tab-btn')[0].classList.add('active');fetchData()}

/* ===== INSIDER TRADING SCREENER (OPENINSIDER) ===== */
async function runInsiderScreener(){
    const btn=document.getElementById('insider-screener-btn');
    btn.querySelector('.btn-text').textContent='searching...';btn.querySelector('.btn-loader').classList.remove('hidden');btn.disabled=true;

    const tradeType=document.getElementById('insider-type').value;
    const minValue=parseInt(document.getElementById('insider-min-value').value);
    const days=parseInt(document.getElementById('insider-days').value);
    const ceoCfo=document.getElementById('insider-who').value==='true';

    try{
        const res=await fetch('/api/insider_screener',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({trade_type:tradeType,min_value:minValue,days:days,ceo_cfo_only:ceoCfo,count:100})});
        const data=await res.json();

        btn.querySelector('.btn-text').textContent='search trades';btn.querySelector('.btn-loader').classList.add('hidden');btn.disabled=false;

        if(data.error&&!data.trades.length){showStatus('error: '+data.error,'error');return}

        document.getElementById('insider-screener-results').classList.remove('hidden');
        document.getElementById('insider-results-title').textContent=`${data.count} insider trades found`;

        const container=document.getElementById('insider-screener-table');
        if(!data.trades.length){container.innerHTML='<p class="no-trades">no trades found matching your filters. try lowering the min value or extending the lookback period.</p>';return}

        let h=`<table class="screener-table insider-screener-table"><thead><tr><th>trade date</th><th>ticker</th><th>insider</th><th>title</th><th>type</th><th>price</th><th>qty</th><th>value</th></tr></thead><tbody>`;

        data.trades.forEach(t=>{
            const ac=t.action==='buy'?'var(--green)':'var(--red)';
            h+=`<tr>
                <td>${t.trade_date}</td>
                <td><span class="ticker-link" onclick="loadFromScreener('${t.ticker}')">${t.ticker}</span></td>
                <td>${t.insider_name}</td>
                <td style="color:var(--text-muted);font-size:10px">${t.title}</td>
                <td><span class="action-tag ${t.action}">${t.action}</span></td>
                <td>$${t.price}</td>
                <td>${t.qty?t.qty.toLocaleString():'-'}</td>
                <td class="value-cell ${t.action}" style="color:${ac};font-weight:700">${fmtVal(t.value)}</td>
            </tr>`;
        });

        h+='</tbody></table>';container.innerHTML=h;

    }catch(e){
        btn.querySelector('.btn-text').textContent='search trades';btn.querySelector('.btn-loader').classList.add('hidden');btn.disabled=false;
        showStatus('error: '+e.message,'error');
    }
}

/* ===== TRADES ===== */
async function saveTrade(td,btn){try{const r=await fetch('/api/save_trade',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(td)});if(r.ok){btn.textContent='saved!';btn.classList.add('saved');loadTrades()}}catch(e){}}
async function loadTrades(){try{const trades=await(await fetch('/api/trades')).json();const table=document.getElementById('trades-table'),nm=document.getElementById('no-trades-msg'),tb=document.getElementById('trades-body');
if(!trades.length){table.classList.add('hidden');nm.classList.remove('hidden');return}table.classList.remove('hidden');nm.classList.add('hidden');
tb.innerHTML=trades.map(t=>{const c=(t.action==='buy'||t.action==='buy_then_sell')?'var(--green)':(t.action==='sell'||t.action==='sell_then_buy')?'var(--red)':'var(--amber)';
return`<tr><td style="color:var(--accent);font-weight:600">${t.ticker}</td><td style="color:${c};font-weight:600;text-transform:uppercase;font-size:11px">${t.action}</td><td>${t.buy_date}</td><td>${t.buy_price!=='-'?'$'+t.buy_price:'-'}</td><td>${t.sell_date}</td><td>${t.sell_price!=='-'?'$'+t.sell_price:'-'}</td><td style="color:var(--green)">${t.potential_gain}</td><td>${t.saved_at||'-'}</td></tr>`}).join('')}catch(e){}}
async function clearTrades(){try{await fetch('/api/clear_trades',{method:'POST'});loadTrades();showStatus('cleared','success')}catch(e){}}
loadTrades();
