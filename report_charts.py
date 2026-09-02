'''
report_charts.py
chart helpers for the pdf report tab

two charts only, both rendered straight to a base64 png so the finished pdf
is self contained. a report found in a folder in six months must still draw
its own pictures with no network and no cache behind it.

    price_chart_png   one year of daily closes with the 50 and 200 day averages
    factor_chart_png  one horizontal bar per module on the -1 to +1 scale

matplotlib is pinned to the Agg backend because these are drawn inside a
flask background thread, where any interactive backend would try to open a
window and take the whole process down with it.
'''

import base64
import io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import numpy as np


#merlin's palette, taken from static/css/style.css so the charts sit in the
#same family as the rest of the app. the pdf page itself is light rather than
#matt black - a research note has to survive being printed
GREEN  = '#00a93a'
RED    = '#f06070'
AMBER  = '#f0a030'
VIOLET = '#a07cff'
ORANGE = '#f97316'
INK    = '#1a1a1a'
MUTED  = '#8a8a8a'
GRID   = '#e2e2e2'

def _resolve_font():
    '''
    pick the first monospace family this machine actually has.

    handing matplotlib a list of families makes it warn once per miss per
    text object, which on a chart with thirty labels is thirty warnings in
    the console for a chart that came out fine. resolve it once instead.
    '''
    try:
        from matplotlib import font_manager
        available = {f.name for f in font_manager.fontManager.ttflist}
        for name in ('JetBrains Mono', 'DejaVu Sans Mono', 'Consolas', 'Courier New'):
            if name in available:
                return name
    except Exception:
        pass
    return 'monospace'


FONT_STACK = _resolve_font()


def _to_base64(fig, dpi=150):
    '''png bytes as a data uri, then close the figure so the thread leaks nothing'''
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode('ascii')


def _blank(message, width=9.0, height=2.4):
    '''placeholder panel so a missing chart never leaves a hole in the layout'''
    fig, ax = plt.subplots(figsize=(width, height))
    ax.text(0.5, 0.5, message, ha='center', va='center',
            fontsize=10, color=MUTED, fontfamily=FONT_STACK)
    ax.axis('off')
    return _to_base64(fig)


def price_chart_png(frame, ticker, currency='USD'):
    '''
    one year of daily closes with the 50 and 200 day moving averages.

    the frame is whatever the report context already fetched, so this never
    touches the network. it expects a datetime index and a Close column and
    computes the two averages itself off the full frame, so the 200 day line
    is real rather than a truncated one built from the visible year alone.
    '''
    try:
        if frame is None or len(frame) < 30:
            return _blank('not enough price history for a chart')

        full = frame['Close'].astype(float)
        df = frame.tail(260)
        close = full.tail(260)
        sma50 = full.rolling(50, min_periods=10).mean().tail(260)
        sma200 = full.rolling(200, min_periods=40).mean().tail(260)

        fig, ax = plt.subplots(figsize=(9.0, 2.35))
        ax.plot(df.index, close, color=INK, linewidth=1.4, label='close', zorder=3)
        ax.plot(df.index, sma50, color=GREEN, linewidth=1.1, label='50 day', zorder=2)
        ax.plot(df.index, sma200, color=VIOLET, linewidth=1.1, label='200 day', zorder=2)
        ax.fill_between(df.index, float(close.min()) * 0.985, close,
                        color=GREEN, alpha=0.05, zorder=1)

        ax.set_ylim(float(close.min()) * 0.97, float(close.max()) * 1.03)
        ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(GRID)

        sym = {'GBp': 'p', 'GBP': '£', 'USD': '$', 'EUR': '€'}.get(currency, '')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{sym}{v:,.0f}'))
        ax.tick_params(labelsize=8, colors=MUTED)
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontfamily(FONT_STACK)

        leg = ax.legend(loc='upper left', fontsize=8, frameon=False, ncol=3)
        for txt in leg.get_texts():
            txt.set_fontfamily(FONT_STACK)
            txt.set_color(MUTED)
        ax.set_title(f'{ticker} - one year daily close', fontsize=10, color=INK,
                     fontfamily=FONT_STACK, loc='left', pad=10)
        return _to_base64(fig)
    except Exception as e:
        return _blank(f'price chart unavailable ({type(e).__name__})')


def factor_chart_png(rows):
    '''
    one horizontal bar per module on the -1 to +1 scale with a zero line.

    this is the picture that makes disagreement obvious at a glance, so
    modules with no score are still drawn - as a hollow grey marker on the
    zero line labelled n/a, never silently dropped.

    rows is a list of dicts with label and score, score being None where the
    module could not answer.
    '''
    try:
        if not rows:
            return _blank('no modules returned a score')

        labels = [r['label'] for r in rows]
        scores = [(r['score'] if r.get('score') is not None else 0.0) for r in rows]
        live = [r.get('score') is not None for r in rows]

        height = max(1.8, 0.21 * len(rows) + 0.8)
        fig, ax = plt.subplots(figsize=(9.0, height))

        ypos = np.arange(len(rows))[::-1]
        colours = []
        for sc, is_live in zip(scores, live):
            if not is_live:
                colours.append('#d8d8d8')
            elif sc > 0.05:
                colours.append(GREEN)
            elif sc < -0.05:
                colours.append(RED)
            else:
                colours.append(AMBER)

        ax.barh(ypos, scores, color=colours, height=0.55, zorder=3)
        for y, sc, is_live in zip(ypos, scores, live):
            if not is_live:
                ax.plot([0], [y], marker='o', markersize=5, markerfacecolor='white',
                        markeredgecolor='#c0c0c0', zorder=4)
                ax.text(0.04, y, 'n/a', va='center', fontsize=7.5,
                        color=MUTED, fontfamily=FONT_STACK, zorder=4)
            else:
                off = 0.035 if sc >= 0 else -0.035
                ax.text(sc + off, y, f'{sc:+.2f}', va='center',
                        ha='left' if sc >= 0 else 'right',
                        fontsize=7.5, color=INK, fontfamily=FONT_STACK, zorder=4)

        ax.axvline(0, color=INK, linewidth=1.0, zorder=2)
        ax.set_xlim(-1.15, 1.15)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.set_xticklabels(['-1.0 avoid', '-0.5', '0', '+0.5', '+1.0 buy'], fontsize=8)
        ax.grid(True, axis='x', color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(GRID)
        ax.tick_params(labelsize=8, colors=MUTED, length=0)
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontfamily(FONT_STACK)
        for lab in ax.get_yticklabels():
            lab.set_color(INK)
        ax.set_title('module scores on the -1 to +1 scale', fontsize=10, color=INK,
                     fontfamily=FONT_STACK, loc='left', pad=10)
        return _to_base64(fig)
    except Exception as e:
        return _blank(f'factor chart unavailable ({type(e).__name__})')
