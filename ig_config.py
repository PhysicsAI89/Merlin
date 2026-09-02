'''
IG configuration, shared by app.py and build_ig_epics.py.

the only secrets merlin has are the IG credentials cortex uses for live
prices. they are read from a .env file in the project root so they never end
up in the source. deliberately hand rolled rather than pulling in
python-dotenv: it is twenty lines and keeps requirements.txt short.

this lives in its own module because two entry points need it before they can
talk to IG, and importing app.py from a helper script would drag in keras and
the whole flask app for the sake of five lines.

nothing here is required to run merlin. with no .env, cortex falls back to
delayed yahoo prices and every other tab is unaffected.
'''

import os

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')

#IG runs two completely separate platforms, each with its own credentials,
#its own API key and its own account numbers. demo is the default because a
#wrong epic on a live account is a real order, and snapshot prices are
#identical on both.
IG_HOSTS = {'demo': 'https://demo-api.ig.com/gateway/deal',
            'live': 'https://api.ig.com/gateway/deal'}


def load_env_file(path=ENV_PATH):
    '''
    read KEY=VALUE lines into os.environ. returns the number of keys set.

    blank lines and # comments are skipped, surrounding quotes are stripped,
    and an unreadable file is not fatal - IG is optional and cortex falls back
    to yahoo whenever the credentials are missing.

    a real environment variable always wins over the file, so a value exported
    in the shell can override .env without editing it.
    '''
    if not os.path.exists(path):
        return 0
    n = 0
    try:
        #utf-8-sig because notepad on windows writes a BOM, and a BOM on the
        #first line turns IG_API_KEY into ﻿IG_API_KEY, which then silently
        #does not match anything
        with open(path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, _, val = line.partition('=')
                key, val = key.strip(), val.strip()
                #an inline comment only counts as one when it follows
                #whitespace, so a # inside an unquoted password survives
                if not (val.startswith('"') or val.startswith("'")):
                    for i, ch in enumerate(val):
                        if ch == '#' and i > 0 and val[i - 1].isspace():
                            val = val[:i].rstrip()
                            break
                if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                    val = val[1:-1]
                if key and key not in os.environ:
                    os.environ[key] = val
                    n += 1
    except Exception as e:
        print(f'could not read {path}: {type(e).__name__}: {e}\n')
    return n


def ig_env():
    '''demo or live, defaulting to demo when unset or misspelled'''
    env = (os.getenv('IG_ENV') or 'demo').strip().lower()
    if env not in IG_HOSTS:
        print(f'IG_ENV={env!r} is not demo or live, falling back to demo\n')
        env = 'demo'
    return env


def configure_cortex(cortex, verbose=True):
    '''
    point cortex at the right IG host and say what state IG is in.

    set here rather than edited into cortex.py so that module stays free of
    merlin's configuration, the same reason the US universe is injected.
    reassigning the module attribute works because cortex reads IG_BASE_URL
    at call time, never captures it at import.
    '''
    env = ig_env()
    cortex.IG_BASE_URL = IG_HOSTS[env]
    if not verbose:
        return env
    if cortex.ig_configured():
        n = len(cortex.load_ig_epics())
        tail = '' if n else ' - run build_ig_epics.py to map them'
        print(f'IG credentials found, {env} platform, {n} epics mapped{tail}\n')
    else:
        print('no IG credentials, cortex will use delayed yahoo prices\n')
    return env
