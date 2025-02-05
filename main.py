import typing as t
import json

import multiprocessing
import networkx as nx


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
                target_dict = {k: v for k, v in sorted(_target_dict.items(), key=lambda j: j[1])}
                all_stations = [start_station] + list(target_dict.keys())
                times = {}
                for i in range(len(all_stations) - 1):
                    first_station = all_stations[i]
                    target_station = all_stations[i + 1]
                    times[(first_station, target_station)] = float(target_dict[target_station] - target_dict.get(first_station, 0))

                for (first_station, second_station), travel_time in times.items():
                    G.add_edge(
                        first_station,
                        second_station,
                        weight=travel_time,
                        lines={line_id}
                        )
    return G


def tsp_nearest_neighbor(G: nx.Graph, start: str) -> t.List[str]:
    unvisited = set(G.nodes())
    order = [start]
    unvisited.remove(start)
    current = start

    while unvisited:
        next_station = min(unvisited, key=lambda x: nx.dijkstra_path_length(G, current, x, weight='weight'))
        order.append(next_station)
        unvisited.remove(next_station)
        current = next_station

    order.append(start)
    return order


def full_tsp_path(G: nx.Graph, tsp_order: t.List[str]) -> t.List[str]:
    full_route = []
    for i in range(len(tsp_order) - 1):
        leg = nx.dijkstra_path(G, tsp_order[i], tsp_order[i + 1], weight='weight')
        if i > 0:
            leg = leg[1:]
        full_route.extend(leg)
    return full_route


def tsp_main(g, start_station):
    tsp_order = tsp_nearest_neighbor(g, start_station)
    full_route = full_tsp_path(g, tsp_order)
    total_time = 0
    for i in range(len(tsp_order) - 1):
        leg_time = nx.dijkstra_path_length(g, tsp_order[i], tsp_order[i + 1], weight='weight')
        total_time += leg_time

    return {'start_station': start_station, 'route': full_route, 'time': total_time}


def main():
    lines, stations, line_routes = build_network()
    g = build_graph(stations, raw_intervals)

    with multiprocessing.Pool() as pool:
        results = pool.starmap(tsp_main, [(g, station) for station in stations])

    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    main()
