from pathlib import Path
import json
import time

import requests


lines_path = Path('lines.json')
if lines_path.exists():
    with open(lines_path, 'r') as f:
        lines = json.load(f)
else:
    r = requests.get('https://api.tfl.gov.uk/Line/Route')
    res = r.json()
    lines = {}
    for line in res:
        if line['modeName'] != 'tube':
            continue
        rs = [i for i in line['routeSections'] if i['direction'] == 'outbound']
        lines[line['id']] = {
            'name': line['name'],
            'routes': [{'originator': i['originator'], 'destination': i['destination']}
                       for i in line['routeSections'] if i['direction'] == 'outbound'],
        }
    with open(lines_path, 'w') as f:
        json.dump(lines, f, indent=4)

stops_path = Path('stops.json')
if stops_path.exists():
    with open(stops_path, 'r') as f:
        stops = json.load(f)
else:
    stops = {}
    for line in lines:
        r = requests.get(f'https://api.tfl.gov.uk/Line/{line}/Route/Sequence/outbound')
        res = r.json()
        ordered_stops = {i: [s['stationId'] for s in sps['stopPoint']] for i, sps in enumerate(res['stopPointSequences'])}
        stops[line] = ordered_stops

    with open(stops_path, 'w') as f:
        json.dump(stops, f, indent=4)


def get_timetable(line, start, end):
    max_retries = 5
    while True:
        r = requests.get(
            f'https://api.tfl.gov.uk/Line/{line}/Timetable/{start}/to/{end}'
        )
        if r.status_code == 200:
            return r.json()
        else:
            max_retries -= 1
            if max_retries == 0:
                return None
            print(f'{r.status_code} - Retrying in 10 seconds...')
            time.sleep(10)

stations_path = Path('stations.json')
intervals_path = Path('intervals.json')

if not (stations_path.exists() and intervals_path.exists()):
    stations = {}
    intervals = {}

    for line in stops:
        intervals[line] = {}
        for branch in stops[line]:
            intervals[line][branch] = {}
            res = get_timetable(line, stops[line][branch][0], stops[line][branch][-1])
            if res is None:
                raise Exception('')

            for station in res['stations']:
                if 'tube' not in station['modes']:
                    continue
                stations[station['id']] = {
                    'name': station['name'],
                    'lines': [l['id'] for l in station['lines'] if l['id'] in lines]
                }

            intervals[line][branch][stops[line][branch][0]] = {}
            for route in res['timetable']['routes']:
                for s_interval in route['stationIntervals']:
                    for interval in s_interval['intervals']:
                        intervals[line][branch][stops[line][branch][0]][interval['stopId']] = interval['timeToArrival']

    with open(stations_path, 'w') as f:
        json.dump(stations, f, indent=4)

    with open(intervals_path, 'w') as f:
        json.dump(intervals, f, indent=4)
