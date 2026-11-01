"""Compact view of a recorded game: one or two lines per turn. usage: view2.py <file.json>"""
import io
import json
import re
import sys

RATING = re.compile(r"\|raw\|(.*?)'s rating: (\d+) (?:&rarr;|→) <strong>(\d+)</strong>")


def view(path):
    r = json.load(io.open(path, encoding='utf-8'))
    side = r['me']['side']
    opp_side = 'p1' if side == 'p2' else 'p2'
    out = []

    def tag(s):
        return s.replace(side + ' ', 'ME ').replace(opp_side + ' ', 'OP ')

    opp_elo = None
    for ln in r['log']:
        m = RATING.search(ln)
        if m and m.group(1).lower().replace(' ', '') != r['me']['name'].lower().replace(' ', ''):
            opp_elo = m.group(2)
    out.append('%s | %s | %d turns | my elo %s | opp elo %s | bot %s %s' % (
        r['id'][-12:], r['result'].upper(), r['turns'], (r.get('elo') or {}).get('before'), opp_elo, r.get('bot'), r.get('build', '')))
    for p in r['myTeam'] or []:
        out.append('  MY %-22s %-15s %-14s tera %-8s %s' % (p['details'].replace(', M', '').replace(', F', ''), p.get('item', ''), p.get('baseAbility', ''), p.get('teraType', ''), '/'.join(p['moves'])))
    out.append('  OPP ' + ' ; '.join('%s L%s%s%s [%s]' % (p['species'], p['level'], (' @' + (p['item'] or p['prevItem'])) if (p['item'] or p['prevItem']) else '', (' tera ' + p['tera']) if p['tera'] else '', ','.join(m['move'] for m in p['moves'])) for p in r['oppTeam']))

    by_turn = {}
    for d in r['decisions']:
        by_turn.setdefault(d['turn'], []).append(d)
    for h in r['history']:
        t = h['turn']
        for d in by_turn.get(t, []):
            me = next((m for m in d['mine'] if m['active']), None)
            op = next((o for o in d['opp'] if o['active']), None)
            opts = ' | '.join('%s %.2f%s' % (o['label'].replace('switch ', '>'), o['v'] if o['v'] is not None else 0,
                                              ('(k%.0f d%.0f)' % (100 * (o['ko'] or 0), 100 * (o['dead'] or 0))) if o['ko'] is not None else '') for o in d['options'][:4])
            out.append('T%-2d %s%s %s vs %s %s%%%s => %s   [%s]' % (
                t, 'F ' if d['forced'] else '', me['species'] if me else '?', me['condition'] if me else '', op['species'] if op else '?',
                op['hp'] if op else '?', ' ' + op['status'] if op and op['status'] else '', d['why'].replace('switch ', '>'), opts))
        ev = [tag(e) for e in h['events']]
        short = []
        for e in ev:
            s = e.strip()
            if e.startswith('  '):
                if 'down to' in s:
                    short.append(s.replace(' down to ', '>').replace(' [from] item: ', ' ').replace(' [from] ', ' '))
                elif ' +' in s or ' -' in s or 'got ' in s or 'Terastallized' in s or s in ('crit', 'miss'):
                    short.append(s)
            else:
                short.append(s.replace(' used ', ': ').replace(' sent in ', ' IN '))
        if short:
            out.append('      ' + '; '.join(short)[:420])
    return '\n'.join(out)


if __name__ == '__main__':
    print(view(sys.argv[1]))
