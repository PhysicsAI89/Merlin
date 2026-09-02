'''
reprice the cortex paper book onto the spread bet model.

why this exists. the book has three closed trades that were settled under
the old share dealing cost model - a flat percentage on a £100 notional,
no leverage. from 10.0 the book is an IG spread bet at five to one, £100
down controlling £500, and the costs are the dealing spread plus overnight
funding. leaving the two side by side in one equity curve means the curve
is measuring two different products and the P/L means nothing.

this rewrites every closed trade's cost, net return and pounds using the
current model in cortex.py. it does NOT touch the entry, the exit, the
prices or the gross return - the trade happened, only what it cost changes.

    python reprice_cortex_book.py            what would change, no writes
    python reprice_cortex_book.py --apply    take a backup and write it

nothing is written without --apply, and --apply copies the book to
data/cortex_paper.backup.json first.
'''
import json
import os
import shutil
import sys

import net_trust
net_trust.install()

import cortex

PATH = cortex.PAPER_PATH
BACKUP = os.path.join(cortex.DATA_DIR, 'cortex_paper.backup.json')


def main():
    apply = '--apply' in sys.argv
    book = cortex.load_paper()
    closed = book.get('closed') or []
    if not closed:
        print('\nnothing closed to reprice')
        return

    print('\n%-6s %-6s %-4s %8s %9s %9s %9s %9s' %
          ('', 'side', 'mkt', 'gross', 'old cost', 'new cost', 'old P/L', 'new P/L'))
    old_total = new_total = 0.0
    updated = []
    for t in closed:
        gross = float(t.get('gross_return_pct', 0) or 0)
        nights = int(t.get('nights_held') or
                     cortex._nights_between(t.get('entry_date'), t.get('exit_date')))
        cost = cortex.round_trip_cost_pct(t.get('side'), t.get('market'), nights,
                                          t.get('spread_pct'))
        net = round(gross - cost, 3)
        pl = round(cortex.pl_gbp_for(net, float(t.get('stake_gbp', cortex.STAKE_GBP))), 2)
        old_pl = float(t.get('pl_gbp', 0) or 0)
        old_total += old_pl
        new_total += pl
        print('%-6s %-6s %-4s %+7.3f%% %8.3f%% %8.3f%% %+8.2f %+8.2f' %
              (t.get('ticker'), t.get('side'), t.get('market'), gross,
               float(t.get('cost_pct', 0) or 0), cost, old_pl, pl))
        rec = dict(t)
        rec.update({
            'cost_pct': cost, 'return_pct': net, 'pl_gbp': pl,
            'nights_held': nights,
            'return_on_stake_pct': round(net * cortex.CORTEX_LEVERAGE, 3),
            'spread_pct_charged': round(cortex.spread_cost_pct(t.get('market'),
                                                               t.get('spread_pct')), 4),
            'funding_pct': round(cortex.funding_cost_pct(t.get('side'), t.get('market'),
                                                         nights), 4),
            'exposure_gbp': round(float(t.get('stake_gbp', cortex.STAKE_GBP))
                                  * cortex.CORTEX_LEVERAGE, 2),
            'leverage': cortex.CORTEX_LEVERAGE,
            'repriced': 'spread bet model, 10.0',
        })
        rec.pop('financing_pct', None)
        updated.append(rec)

    print('\n%-24s %+8.2f -> %+8.2f' % ('realised P/L, pounds', old_total, new_total))
    print('%-24s %s' % ('leverage applied', '%gx' % cortex.CORTEX_LEVERAGE))

    if not apply:
        print('\ndry run. nothing written. re-run with --apply to take a backup and commit')
        return

    shutil.copy(PATH, BACKUP)
    book['closed'] = updated
    book['realised_pl_gbp'] = round(new_total, 4)
    #fees follow the new cost model too, or the counter stops agreeing with
    #the P/L it is supposed to explain
    book['fees_paid_gbp'] = round(sum(
        cortex.pl_gbp_for(float(t['cost_pct']), float(t.get('stake_gbp', cortex.STAKE_GBP)))
        for t in updated), 4)
    cortex._rebuild_equity(book)
    cortex.save_paper(book)
    print('\nbacked up to %s' % BACKUP)
    print('written. %d trades repriced, realised P/L now £%.2f' % (len(updated), new_total))


if __name__ == '__main__':
    main()
