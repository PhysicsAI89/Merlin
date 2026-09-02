'''
build data/cortex_ig_epics.csv, the ticker -> IG epic map cortex needs before
it can serve live prices.

cortex knows FTSE 350 tickers. IG knows epics, its own instrument ids, which
look like KA.D.VOD.CASH.IP. nothing in either system translates between the
two, so this walks the universe once, searches IG for each ticker and writes
the pairs out. it is a setup step, not part of a scan: run it once, check the
result, and only re-run it when the universe changes.

usage:

    python build_ig_epics.py --test       check credentials, list accounts
    python build_ig_epics.py              map every LSE ticker (about 13 min)
    python build_ig_epics.py --limit 20   map the first 20 only, for a dry run
    python build_ig_epics.py --refresh    remap tickers already in the file
    python build_ig_epics.py --from-dump  redo the matching offline, no API calls
    python build_ig_epics.py --browse     dump every UK share epic IG offers

three files come out of a full run:

    data/cortex_ig_epics.csv          confident matches, this is what cortex reads
    data/cortex_ig_epics_review.csv   everything ambiguous or unmatched
    data/ig_search_dump.json          every raw search result, for --from-dump

only the first is ever used for live prices. anything the matcher was not
sure about is parked in the review file instead of quietly becoming the price
source for a real trade - move a row across by hand once you have checked the
epic on the IG platform. the review file earns its keep: ABDN's real listing
is KA.D.SLLN.CASH.IP, still carrying Standard Life's old code, which no
automatic rule should be confident about.

matching is deliberately strict, and needs both the instrument code and the
exchange to agree. see confidence_for() for why either alone gets you the
wrong company.
'''

import argparse
import csv
import json
import os
import sys
import time

import ig_config

#credentials must be in the environment before cortex is asked anything about
#IG, so load .env first
ig_config.load_env_file()

import cortex

IG_ENV = ig_config.configure_cortex(cortex, verbose=False)

REVIEW_PATH = os.path.join('data', 'cortex_ig_epics_review.csv')
#every search result from the last run, so the matching rules can be
#changed and re-applied with --from-dump instead of re-querying IG
DUMP_PATH   = os.path.join('data', 'ig_search_dump.json')

#IG allows 30 non-trading requests a minute per app key. one search is one
#request, so 2.2 seconds between them sits just under the limit with a little
#headroom for retries
DEFAULT_GAP_S = 2.2

#instruments that share a ticker with the share we want but are not it
BAD_NAME_BITS = ('ADR', 'ADS', 'OPTION', 'FUTURE', 'WARRANT', 'BASKET',
                 'INDEX', 'KNOCKOUT', 'TURBO')

#an epic is EXCHANGE.D.CODE.TYPE.IP, e.g. KA.D.VOD.CASH.IP.
#
#the exchange field is what stops a right code on the wrong market being taken
#for a match, and ticker codes are only unique per exchange, so this matters:
#ALFA alone matches EH.D.ALFA (Alfa Laval AB, Stockholm) just as well as it
#matches the London listing. other exchanges seen in testing are AR
#(Johannesburg), UD (US ADRs), ED (Germany), EA (Belgium), AC (Hong Kong) and
#SI (US ETNs).
#
#london appears under more than one prefix - KA for most of it, KC for a lot
#of the investment trusts - so this is a set rather than one value.
LSE_EPIC_PREFIXES = ('KA', 'KC', 'KB')

#IG suffixes many london codes with LN, bloomberg style: ALFA is ALFALN, AIE
#is AIELN. plenty of others carry the bare ticker (VOD, BARC, HSBA), so both
#forms have to be accepted, and the suffix has to come off before comparing or
#every suffixed code reads as a mismatch.
LSE_CODE_SUFFIX = 'LN'


def ig_search_code(ticker):
    '''
    the code IG is likely to use for a cortex ticker.

    cortex stores BT.A as BT-A because that is what yahoo wants. IG strips the
    punctuation instead, so both forms are worth trying.
    '''
    return ticker.replace('-', '.'), ticker.replace('-', '').replace('.', '')


def epic_parts(epic):
    '''(exchange, code) from an epic, both upper case, blanks if malformed'''
    bits = (epic or '').split('.')
    if len(bits) < 3:
        return '', ''
    return bits[0].upper(), bits[2].upper()


def code_matches(ticker, code):
    '''
    does this epic code name our ticker, allowing for the LN london suffix.

    both KA.D.VOD (bare) and KA.D.ALFALN (suffixed) are legitimate, so the
    suffix is stripped before comparing. this is an exact test either way -
    a prefix or substring test would let ALBKLN match AJB, which is AIB Group
    being mistaken for AJ Bell.
    '''
    if not code:
        return False
    dotted, squashed = ig_search_code(ticker)
    wanted = {dotted.upper(), squashed.upper(), ticker.upper()}
    bare = code[:-len(LSE_CODE_SUFFIX)] if code.endswith(LSE_CODE_SUFFIX) else code
    return code in wanted or bare in wanted


def score_match(ticker, market):
    '''
    how strongly an IG search result looks like the LSE share we asked for.

    returns a tuple starting with the score so a list of these sorts best
    first. the two signals that carry real weight are the instrument code and
    the exchange; type and expiry only separate an ordinary share from the
    options and futures written on it.
    '''
    epic = (market.get('epic') or '').strip()
    name = (market.get('instrumentName') or '').upper()
    itype = (market.get('instrumentType') or '').upper()
    expiry = (market.get('expiry') or '').strip()
    exchange, code = epic_parts(epic)

    score = 0
    if code_matches(ticker, code):
        score += 8
    if itype == 'SHARES':
        score += 4
    if expiry in ('-', 'DFB', ''):
        score += 3
    if epic.endswith('.IP'):
        score += 2
    #the exchange outranks everything above: a right code on the wrong market
    #is a different company, not a near miss
    if exchange in LSE_EPIC_PREFIXES:
        score += 6
    else:
        score -= 8
    for bit in BAD_NAME_BITS:
        if bit in name:
            score -= 6
            break
    return score, epic, market.get('instrumentName') or '', itype, expiry


def confidence_for(score, code_matched, itype, on_lse):
    '''
    high and medium are written to the live map, low never is.

    both tiers require two things: the epic's instrument code equals the
    ticker, and the listing is on the LSE. either alone is not enough.

    an IG search for BARC also returns Barco NV and two GraniteShares 3x
    Barclays ETPs, which score respectably on type and expiry, so the code
    test throws those out. and a code match on its own still is not evidence
    of the right company, because codes only have to be unique per exchange -
    ALFA matches Alfa Laval in Stockholm perfectly. the exchange test throws
    those out.

    a ticker that fails either test goes to the review file and falls back to
    yahoo, which is a supported state and much cheaper than pricing a signal
    off the wrong company.
    '''
    if not code_matched or not on_lse:
        return 'low'
    if itype.upper() == 'SHARES' and score >= 19:
        return 'high'
    if score >= 15:
        return 'medium'
    return 'low'


def search(ig, term, gap_s):
    '''
    one /markets?searchTerm= call.

    cortex.IGClient covers login, throttling and logout but only ever needs
    snapshots, so the search endpoint is driven through its session here
    rather than added to cortex.py, which is finished and should stay that way.
    '''
    time.sleep(gap_s)
    r = ig.session.get(f'{cortex.IG_BASE_URL}/markets',
                       params={'searchTerm': term},
                       headers=ig._headers('1'), timeout=20)
    r.raise_for_status()
    return (r.json() or {}).get('markets') or []


def load_existing():
    '''ticker -> epic already mapped, so a re-run can resume'''
    out = {}
    path = cortex.IG_EPIC_MAP
    if os.path.exists(path):
        with open(path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                t, e = (row.get('ticker') or '').strip(), (row.get('epic') or '').strip()
                if t and e:
                    out[t] = row
    return out


def write_map(rows):
    '''
    cortex only reads ticker and epic. the rest is there so the file can be
    audited months later without going back to IG
    '''
    os.makedirs(os.path.dirname(cortex.IG_EPIC_MAP) or '.', exist_ok=True)
    cols = ['ticker', 'epic', 'instrument_name', 'instrument_type', 'expiry',
            'confidence', 'mapped_at']
    with open(cortex.IG_EPIC_MAP, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for t in sorted(rows):
            w.writerow({c: rows[t].get(c, '') for c in cols})


def write_review(rows):
    os.makedirs(os.path.dirname(REVIEW_PATH) or '.', exist_ok=True)
    cols = ['ticker', 'epic', 'instrument_name', 'instrument_type', 'expiry',
            'confidence', 'score', 'note']
    with open(REVIEW_PATH, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in cols})


def cmd_test():
    '''
    prove the credentials work before spending fifteen minutes on a mapping run.

    also prints the account ids, because IG_ACCOUNT_ID is the one setting you
    cannot guess and the spread bet and share dealing accounts behave
    differently enough that picking the wrong one is worth catching early.
    '''
    if not cortex.ig_configured():
        print('no IG credentials found.\n')
        print('create a .env file in this folder with:\n')
        print('    IG_API_KEY=your_key')
        print('    IG_USERNAME=your_login')
        print('    IG_PASSWORD=your_password')
        print('    IG_ACCOUNT_ID=ABC12')
        print('    IG_ENV=demo\n')
        return 1

    print(f'platform : {IG_ENV}')
    print(f'host     : {cortex.IG_BASE_URL}')
    print(f'username : {os.getenv("IG_USERNAME")}\n')

    import requests
    r = requests.post(f'{cortex.IG_BASE_URL}/session',
                      json={'identifier': os.getenv('IG_USERNAME'),
                            'password': os.getenv('IG_PASSWORD')},
                      headers={'Content-Type': 'application/json; charset=UTF-8',
                               'Accept': 'application/json; charset=UTF-8',
                               'X-IG-API-KEY': os.getenv('IG_API_KEY'),
                               'Version': '2'}, timeout=20)
    if r.status_code != 200:
        print(f'login failed, HTTP {r.status_code}')
        print(f'{r.text[:400]}\n')
        print(explain_error(r))
        return 1

    body = r.json() or {}
    print('login OK\n')
    print(f'current account : {body.get("currentAccountId")}')
    accounts = body.get('accounts') or []
    if accounts:
        print('accounts on this login:')
        for a in accounts:
            print(f'  {a.get("accountId"):<12} {a.get("accountName","")} '
                  f'({a.get("accountType","")}, {a.get("currency","")})')
    want = os.getenv('IG_ACCOUNT_ID')
    if want and accounts and want not in [a.get('accountId') for a in accounts]:
        print(f'\nwarning: IG_ACCOUNT_ID={want} is not one of the above')
    print('')

    #a search proves the market data permission, which login alone does not
    with cortex.IGClient() as ig:
        found = search(ig, 'VOD', 0.5)
    print(f'search for VOD returned {len(found)} markets')
    for m in found[:5]:
        print(f'  {m.get("epic"):<28} {m.get("instrumentType",""):<10} '
              f'{m.get("instrumentName","")}')
    print('\ncredentials work. run build_ig_epics.py with no arguments to map '
          'the universe.\n')
    return 0


def explain_error(r):
    '''IG error codes are opaque, so translate the ones that actually happen'''
    code = ''
    try:
        code = (r.json() or {}).get('errorCode', '')
    except Exception:
        code = r.text[:200]
    hints = {
        'error.security.invalid-details':
            'wrong username, password or API key. note the demo platform needs '
            'a demo API key generated from a demo account, they are not shared.',
        'error.security.api-key-invalid':
            'the API key is wrong, or belongs to the other platform. demo keys '
            'only work against demo-api.ig.com and live keys only against api.ig.com.',
        'error.security.api-key-disabled':
            'the key exists but is disabled in My IG.',
        'error.public-api.failure.encryption.required':
            'this account requires encrypted logins. generate a fresh API key, '
            'which defaults to allowing plain logins over https.',
        'error.security.account-token-invalid':
            'session expired, run the command again.',
        'error.public-api.exceeded-account-allowance':
            'rate limited. wait a minute and re-run, and raise --gap.',
    }
    return f'IG said: {code}\n{hints.get(code, "")}'.strip()


def cmd_browse(gap_s):
    '''
    walk IG's market navigation tree and dump every UK share it offers.

    the search based mapping below is usually enough, but when a ticker has no
    confident match this file is how you find the epic by eye: search it for
    the company name rather than the ticker.
    '''
    out, seen = [], set()
    with cortex.IGClient() as ig:
        def walk(node_id=None, path='', depth=0):
            if depth > 4:
                return
            url = f'{cortex.IG_BASE_URL}/marketnavigation'
            if node_id:
                url += f'/{node_id}'
            time.sleep(gap_s)
            try:
                r = ig.session.get(url, headers=ig._headers('1'), timeout=20)
                r.raise_for_status()
                body = r.json() or {}
            except Exception as e:
                print(f'  skip {path}: {type(e).__name__}')
                return
            for m in (body.get('markets') or []):
                epic = m.get('epic')
                if epic and epic not in seen:
                    seen.add(epic)
                    out.append({'epic': epic,
                                'instrument_name': m.get('instrumentName', ''),
                                'instrument_type': m.get('instrumentType', ''),
                                'expiry': m.get('expiry', ''),
                                'path': path})
            for n in (body.get('nodes') or []):
                name = (n.get('name') or '')
                #only descend where UK shares can be, the full tree is enormous
                if depth == 0 and not any(k in name.lower() for k in
                                          ('share', 'equit', 'uk', 'ftse')):
                    continue
                walk(n.get('id'), f'{path}/{name}', depth + 1)
            print(f'  {len(out)} markets so far ({path or "root"})')

        walk()

    path = os.path.join('data', 'ig_uk_markets.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['epic', 'instrument_name',
                                          'instrument_type', 'expiry', 'path'])
        w.writeheader()
        w.writerows(out)
    print(f'\nwrote {len(out)} markets to {path}\n')
    return 0


def cmd_build(limit, refresh, gap_s, from_dump=False):
    '''
    the mapping run.

    every search result is kept in DUMP_PATH, not just the winner. matching
    conventions are guesswork until you have seen real data - the LN suffix
    and the KC exchange prefix both turned up only once a run was under way -
    and re-tuning against a saved dump takes a second, where re-querying IG
    takes a quarter of an hour. --from-dump rebuilds the map offline.
    '''
    if from_dump:
        if not os.path.exists(DUMP_PATH):
            print(f'no dump at {DUMP_PATH}. run a normal build first.\n')
            return 1
        with open(DUMP_PATH, encoding='utf-8') as f:
            dump = json.load(f)
        print(f'rebuilding from {DUMP_PATH}, {len(dump)} tickers, no API calls\n')
        return build_from_results(dump, {})

    if not cortex.ig_configured():
        print('no IG credentials. run with --test for the setup instructions.\n')
        return 1

    tickers = list(cortex.LSE_UNIVERSE)
    if limit:
        tickers = tickers[:limit]

    existing = {} if refresh else load_existing()
    todo = [t for t in tickers if t not in existing]

    print(f'universe   : {len(tickers)} LSE tickers')
    print(f'already in : {len(existing)}')
    print(f'to map     : {len(todo)}')
    if not todo:
        print('\nnothing to do. use --refresh to remap everything.\n')
        return 0
    mins = (len(todo) * gap_s) / 60.0
    print(f'estimated  : {mins:.0f} minutes at {gap_s}s between calls\n')

    #seed from the previous dump so a resumed run accumulates rather than
    #replacing it. without this, a run that skipped 300 already-mapped tickers
    #would leave a dump holding only the 49 it just did, and --from-dump would
    #quietly rebuild a map missing everything else
    raw = load_dump()
    failed = []

    with cortex.IGClient() as ig:
        for i, ticker in enumerate(todo, 1):
            dotted, squashed = ig_search_code(ticker)
            results = []
            for term in ([dotted] if dotted == squashed else [dotted, squashed]):
                try:
                    results = search(ig, term, gap_s)
                except Exception as e:
                    print(f'[{i}/{len(todo)}] {ticker:<8} search failed: '
                          f'{type(e).__name__}: {str(e)[:80]}')
                    failed.append(ticker)
                    results = []
                    break
                if results:
                    break

            #keep only the fields the matcher looks at, so the dump stays small
            raw[ticker] = [{'epic': m.get('epic'),
                            'instrumentName': m.get('instrumentName'),
                            'instrumentType': m.get('instrumentType'),
                            'expiry': m.get('expiry')} for m in results]

            best = ''
            if results:
                sc = sorted((score_match(ticker, m) for m in results),
                            key=lambda x: x[0], reverse=True)[0]
                best = f'{sc[1]:<28} {(sc[2] or "")[:38]}'
            print(f'[{i}/{len(todo)}] {ticker:<8} {len(results):>2} hits  {best}')

            #written as we go so a dropped connection does not lose the run
            if i % 10 == 0:
                save_dump(raw)

    save_dump(raw)
    if failed:
        print(f'\n{len(failed)} searches errored, re-run to retry them')
    return build_from_results(raw, existing)


def load_dump():
    '''previous run's raw search results, empty dict if there is no dump yet'''
    if not os.path.exists(DUMP_PATH):
        return {}
    try:
        with open(DUMP_PATH, encoding='utf-8') as f:
            return json.load(f) or {}
    except Exception:
        return {}


def save_dump(raw):
    os.makedirs(os.path.dirname(DUMP_PATH) or '.', exist_ok=True)
    with open(DUMP_PATH, 'w', encoding='utf-8') as f:
        json.dump(raw, f, indent=1)


def build_from_results(raw, existing):
    '''
    turn saved search results into the live map and the review file.

    pure function of the dump, no network, so the matching rules can be
    changed and re-applied in a second.
    '''
    mapped = dict(existing)
    review = []
    stamp = time.strftime('%Y-%m-%d')
    counts = {'high': 0, 'medium': 0, 'low': 0, 'none': 0}

    for ticker in sorted(raw):
        results = raw[ticker] or []
        if not results:
            counts['none'] += 1
            review.append({'ticker': ticker, 'confidence': 'none',
                           'note': 'no IG market matched this ticker'})
            continue

        scored = sorted((score_match(ticker, m) for m in results),
                        key=lambda x: x[0], reverse=True)
        score, epic, name, itype, expiry = scored[0]
        exchange, code = epic_parts(epic)
        conf = confidence_for(score, code_matches(ticker, code), itype,
                              exchange in LSE_EPIC_PREFIXES)
        counts[conf] += 1

        if conf in ('high', 'medium'):
            mapped[ticker] = {'ticker': ticker, 'epic': epic,
                              'instrument_name': name, 'instrument_type': itype,
                              'expiry': expiry, 'confidence': conf,
                              'mapped_at': stamp}
        else:
            #park the two best guesses so the ticker can be fixed by hand
            for s2, e2, n2, t2, x2 in scored[:2]:
                review.append({'ticker': ticker, 'epic': e2,
                               'instrument_name': n2, 'instrument_type': t2,
                               'expiry': x2, 'confidence': 'low', 'score': s2,
                               'note': 'check on the IG platform before using'})

    write_map(mapped)
    write_review(review)

    high = sum(1 for r in mapped.values() if r.get('confidence') == 'high')
    med = sum(1 for r in mapped.values() if r.get('confidence') == 'medium')
    needs_look = len({r['ticker'] for r in review})
    print('\n' + '=' * 62)
    print(f'live map : {len(mapped)} tickers  ({high} high, {med} medium)')
    print(f'review   : {needs_look} tickers need a look, {counts["none"]} had no IG match')
    print(f'\n  {cortex.IG_EPIC_MAP}')
    print(f'  {REVIEW_PATH}')
    print(f'  {DUMP_PATH}  (re-tune with --from-dump, no API calls)')
    print('\nmedium rows are in the live map. skim the instrument_name column '
          'once,\nit is enough to spot a wrong one.\n')
    return 0


def main():
    ap = argparse.ArgumentParser(description='build the cortex ticker to IG epic map')
    ap.add_argument('--test', action='store_true',
                    help='check credentials and list accounts, map nothing')
    ap.add_argument('--browse', action='store_true',
                    help='dump every UK share epic IG offers, for manual matching')
    ap.add_argument('--limit', type=int, default=0,
                    help='only map the first N tickers')
    ap.add_argument('--refresh', action='store_true',
                    help='remap tickers already in the file')
    ap.add_argument('--from-dump', action='store_true', dest='from_dump',
                    help='rebuild the map from saved results, no API calls')
    ap.add_argument('--gap', type=float, default=DEFAULT_GAP_S,
                    help=f'seconds between IG calls (default {DEFAULT_GAP_S})')
    a = ap.parse_args()

    try:
        if a.test:
            return cmd_test()
        if a.browse:
            return cmd_browse(a.gap)
        return cmd_build(a.limit, a.refresh, a.gap, a.from_dump)
    except KeyboardInterrupt:
        print('\nstopped. progress up to the last save is still in the csv.\n')
        return 130


if __name__ == '__main__':
    sys.exit(main())
