import typing as t
import json
import multiprocessing

import heapq
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
                    second_station = all_stations[i + 1]
                    times[(first_station, second_station)] = float(
                        target_dict[second_station] - target_dict.get(first_station, 0)
                    )

                for (u, v), travel_time in times.items():
                    G.add_edge(u, v, weight=travel_time, lines={line_id})
    return G


def cover_heuristic(current: str, unvisited: t.Set[str], G: nx.Graph) -> float:
    """
    An admissible heuristic:
      - h1: minimum cost from the current node to any unvisited node.
      - h2: cost of a minimum spanning tree (MST) covering the unvisited nodes.
    """
    if not unvisited:
        return 0.0
    # h1: minimum distance from current to any unvisited node
    try:
        h1 = min(nx.dijkstra_path_length(G, current, u, weight='weight') for u in unvisited)
    except nx.NetworkXNoPath:
        h1 = float('inf')

    # h2: MST cost for the unvisited subgraph.
    subG = G.subgraph(unvisited)
    if subG.number_of_nodes() > 0:
        mst = nx.minimum_spanning_tree(subG, weight='weight')
        h2 = sum(data['weight'] for u, v, data in mst.edges(data=True))
    else:
        h2 = 0.0

    return h1 + h2


def tsp_astar_cover(G: nx.Graph, start: str) -> t.Tuple[t.List[str], float]:
    """
    Uses A* search over states of the form (current_station, visited_set) to
    find the route of minimum travel time that visits all stations (nodes).

    The algorithm is allowed to revisit nodes if that leads to a lower overall cost.
    Returns a tuple (route, total_time).
    """
    all_nodes = set(G.nodes())
    # State: (f, cost_so_far, current, visited, route)
    # where f = cost_so_far + heuristic(current, unvisited)
    start_state = (cover_heuristic(start, all_nodes - {start}, G), 0.0, start, frozenset({start}), [start])
    # Priority queue for A*
    frontier = []
    heapq.heappush(frontier, start_state)

    # For pruning: best cost found so far for (current, visited) state.
    best_cost: t.Dict[t.Tuple[str, frozenset], float] = {}

    while frontier:
        f, cost, current, visited, route = heapq.heappop(frontier)

        # Goal test: all stations visited
        if visited == all_nodes:
            # Optionally, you could add the cost to return to the starting station.
            return route, cost

        state_key = (current, visited)
        if state_key in best_cost and best_cost[state_key] <= cost:
            continue
        best_cost[state_key] = cost

        for neighbor in G.neighbors(current):
            travel_time = G[current][neighbor]['weight']
            new_cost = cost + travel_time
            new_visited = visited | {neighbor}  # even if neighbor was visited, union is harmless
            new_route = route + [neighbor]
            h = cover_heuristic(neighbor, all_nodes - new_visited, G)
            heapq.heappush(frontier, (new_cost + h, new_cost, neighbor, new_visited, new_route))

    return [], float('inf')  # No solution found (should not happen if G is connected)


def main():
    lines, stations, line_routes = build_network()
    g = build_graph(stations, raw_intervals)

    with multiprocessing.Pool() as pool:
        results = pool.starmap(tsp_astar_cover, [(g, station) for station in stations])

    results = [
        {
            'start': station,
            'route': route,
            'total_time': total_time
        } for station, (route, total_time) in zip(stations, results)
    ]

    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    main()
