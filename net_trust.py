'''
net_trust.py
make merlin trust the same certificate authorities windows already does

the problem this solves, in one paragraph. yfinance talks to yahoo through
curl_cffi, and curl_cffi verifies certificates against certifi - a fixed list
of public roots shipped inside a python package. antivirus products with
https scanning turned on (avast, kaspersky, eset, bitdefender and friends)
sit in the middle of every tls connection and re-sign it with a locally
generated root. windows trusts that root because the antivirus installed it
there, so browsers are fine, but certifi has never heard of it and every
yfinance call dies with:

    CertificateVerifyError ... unable to get local issuer certificate

which looks exactly like a network outage and is not one. no amount of
retrying fixes it, and it takes out every tab at once.

the fix is to hand certifi's own bundle plus the roots windows already
trusts to whichever http client asks. this adds no trust the machine does
not already grant system-wide - it just stops python being the only thing on
the box that disagrees with the operating system.

    import net_trust
    net_trust.install()

call it before yfinance is imported. on any platform without a readable
system store, or on any failure at all, it does nothing and merlin carries
on exactly as before.
'''

import os
import ssl

BUNDLE_PATH = os.path.join('data', 'ca_bundle.pem')
REFRESH_DAYS = 7

_installed = False
_report = {'installed': False, 'reason': 'not attempted', 'system_certs': 0}


def _system_certs():
    '''DER blobs from the operating system root stores, empty list if unreadable'''
    out = []
    for store in ('ROOT', 'CA'):
        try:
            for cert, encoding, trust in ssl.enum_certificates(store):
                #trust is True for "trusted for everything", or a tuple of
                #purpose OIDs. either way False means explicitly distrusted
                if trust is not False and encoding == 'x509_asn':
                    out.append(cert)
        except (AttributeError, OSError, PermissionError):
            continue
    return out


def _build_bundle(path):
    import certifi
    with open(certifi.where(), 'rb') as f:
        base = f.read()

    seen = set()
    chunks = [base]
    added = 0
    for der in _system_certs():
        if der in seen:
            continue
        seen.add(der)
        try:
            pem = ssl.DER_cert_to_PEM_cert(der)
        except Exception:
            continue
        if pem in base.decode('ascii', 'ignore'):
            continue
        chunks.append(pem.encode('ascii'))
        added += 1

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(b'\n'.join(chunks))
    os.replace(tmp, path)
    return added


def _stale(path):
    if not os.path.exists(path):
        return True
    try:
        import time
        return (time.time() - os.path.getmtime(path)) > REFRESH_DAYS * 86400
    except OSError:
        return True


def install(verbose=True):
    '''
    build the merged bundle if needed and point every http client at it.

    certifi.where is patched as well as the environment variables, because
    curl_cffi asks certifi directly and ignores CURL_CA_BUNDLE.
    '''
    global _installed, _report
    if _installed:
        return _report
    try:
        path = os.path.abspath(BUNDLE_PATH)
        if _stale(path):
            added = _build_bundle(path)
        else:
            added = -1   #reused, count unknown without re-reading

        import certifi
        import certifi.core
        certifi.where = lambda: path
        certifi.core.where = lambda: path
        for var in ('SSL_CERT_FILE', 'REQUESTS_CA_BUNDLE', 'CURL_CA_BUNDLE'):
            os.environ[var] = path

        _installed = True
        _report = {'installed': True, 'reason': 'ok', 'bundle': path,
                   'system_certs': added}
        if verbose:
            extra = f'{added} system roots added' if added >= 0 else 'cached bundle reused'
            print(f'\n[net_trust] certificate bundle in use: {path} ({extra})')
        return _report
    except Exception as e:
        _report = {'installed': False, 'reason': f'{type(e).__name__}: {str(e)[:120]}',
                   'system_certs': 0}
        if verbose:
            print(f'\n[net_trust] left the default certificate bundle alone ({_report["reason"]})')
        return _report


def status():
    return dict(_report)
