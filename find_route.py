import typing as t
import json

import networkx as nx
from networkx.algorithms import approximation as approx


with open('lines.json', 'r') as f:
    raw_lines = json.load(f)
with open('stops.json', 'r') as f:
    raw_stops = json.load(f)
with open('stations.json', 'r') as f:
    raw_stations = json.load(f)
with open('intervals.json', 'r') as f:
    raw_intervals = json.load(f)


class Line:
    def __init__(self, line_id, name):
        self.line_id = line_id
        self.name = name

    def __repr__(self):
        return f'{self.name} ({self.line_id})'


class Station:
    def __init__(self, station_id: str, name: str, lines: t.List[Line] = None):
        self.station_id = station_id
        self.name = name
        self.lines = lines or []

    def __repr__(self):
        return f'{self.name} ({self.station_id})'


class LineRoute:
    def __init__(self, line: Line, route_id: str, stops: t.List[Station] = None):
        self.line = line
        self.route_id = route_id
        self.stops = stops or []

    def __repr__(self):
        return f'{self.line.name} - {self.route_id}'


def build_network():
    lines = {
        line_id: Line(line_id, line_data['name'])
        for line_id, line_data in raw_lines.items()
    }
    stations = {
        station_id: Station(
            station_id,
            station_data['name'],
            lines=[lines[line_id] for line_id in station_data['lines']]
        )
        for station_id, station_data in raw_stations.items()
    }

    line_routes = []
    for line_id in raw_stops:
        for branch in raw_stops[line_id]:
            route = LineRoute(
                lines[line_id],
                branch,
                [stations[stop] for stop in raw_stops[line_id][branch]]
            )
            line_routes.append(route)

    return lines, stations, line_routes


def build_graph(stations: t.Dict[str, Station], raw_intervals: dict) -> nx.Graph:
    G = nx.Graph()
    for station in stations.values():
        G.add_node(station.station_id, station=station)
    for line_id, branches in raw_intervals.items():
        for branch_id, routes in branches.items():
            for start_station, _target_dict in routes.items():
                # Sort the target dict by cumulative time
                target_dict = {k: v for k, v in sorted(_target_dict.items(), key=lambda j: j[1])}
                all_stations = [start_station] + list(target_dict.keys())
                times = {}
                for i in range(len(all_stations) - 1):
                    u = all_stations[i]
                    v = all_stations[i + 1]
                    times[(u, v)] = float(target_dict[v] - target_dict.get(u, 0))
                for (u, v), travel_time in times.items():
                    G.add_edge(u, v, weight=travel_time, lines={line_id})
    return G


def main():
    lines, stations, line_routes = build_network()
    G = build_graph(stations, raw_intervals)

    cycle = approx.traveling_salesman_problem(
        G, cycle=True, method=approx.christofides
    )
    total_time = 0
    for i in range(len(cycle) - 1):
        u, v = cycle[i], cycle[i + 1]
        total_time += G[u][v]['weight']

    res = {'total_time': total_time, 'route': []}
    for i in range(len(cycle) - 1):
        first_node = G.nodes[cycle[i]]
        second_node = G.nodes[cycle[i + 1]]
        res['route'].append({
            'start': first_node['station'].name,
            'end': second_node['station'].name,
            'line': list(set.intersection(set([l.line_id for l in first_node['station'].lines]), set([l.line_id for l in second_node['station'].lines]))),
        })

    with open('results.json', 'w') as f:
        json.dump(res, f, indent=4)


if __name__ == '__main__':
    main()
