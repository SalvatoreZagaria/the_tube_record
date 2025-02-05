import typing as t
import json
import multiprocessing
import heapq
from functools import lru_cache

import networkx as nx

# -----------------------
# Data loading
# -----------------------
with open('lines.json', 'r') as f:
    raw_lines = json.load(f)
with open('stops.json', 'r') as f:
    raw_stops = json.load(f)
with open('stations.json', 'r') as f:
    raw_stations = json.load(f)
with open('intervals.json', 'r') as f:
    raw_intervals = json.load(f)


# -----------------------
# Domain classes
# -----------------------
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
    """Construct network objects from JSON data."""
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
    """
    Build a NetworkX graph from the raw interval data.
    Assumes that on each branch, times are cumulative.
    """
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
                    u = all_stations[i]
                    v = all_stations[i + 1]
                    times[(u, v)] = float(target_dict[v] - target_dict.get(u, 0))
                for (u, v), travel_time in times.items():
                    if G.has_edge(u, v):
                        G[u][v]['lines'].add(line_id)
                    else:
                        G.add_edge(u, v, weight=travel_time, lines={line_id})
    return G


def prepare_search(G: nx.Graph) -> t.Tuple[t.List[str], t.Dict[str, int], t.Dict[int, str], int, t.List[t.List[float]]]:
    """
    Computes a list of station IDs, maps from station_id to index (and vice versa),
    and a distance matrix of all–pairs shortest path distances.
    """
    station_list = list(G.nodes())
    n = len(station_list)
    mapping = {station: i for i, station in enumerate(station_list)}
    inv_mapping = {i: station for station, i in mapping.items()}

    all_pairs = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    dist_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            u = station_list[i]
            v = station_list[j]
            dist_matrix[i][j] = all_pairs[u].get(v, float('inf'))
    return station_list, mapping, inv_mapping, n, dist_matrix


# -----------------------
# Revised search routine allowing revisits
# -----------------------

# Parameters to control the trade-off between speed and precision:
NEIGHBOR_LIMIT = 1  # maximum number of neighbors to expand from each state
BEAM_WIDTH = 10000  # maximum frontier size at periodic pruning steps


def fast_worker_astar_cover_with_revisits(start_station: str,
                                          mapping: t.Dict[str, int],
                                          inv_mapping: t.Dict[int, str],
                                          n: int,
                                          dist_matrix: t.List[t.List[float]],
                                          neighbor_limit: int = NEIGHBOR_LIMIT,
                                          beam_width: int = BEAM_WIDTH
                                          ) -> t.Tuple[str, t.List[str], float]:
    """
    A* search that finds a route covering all stations (i.e. reaching a goal state where
    every station has been visited at least once) while allowing the route to revisit stations.

    The state is (current_node, visited_mask, route_so_far). Moving to a neighbor that has
    not been visited updates the visited_mask; moving to a node already visited leaves the mask unchanged.

    To control the branching factor, we do the following:
      - Always include all unvisited nodes (if not too many).
      - Also allow a limited number of already–visited nodes (which might serve as shortcuts).
    """

    @lru_cache(maxsize=None)
    def mst_cost(mask: int) -> float:
        """Computes a lower–bound cost to connect the unvisited nodes (using a Minimum Spanning Tree)."""
        nodes = [i for i in range(n) if mask & (1 << i)]
        if not nodes:
            return 0.0
        cost_val = 0.0
        in_mst = {nodes[0]}
        remaining = set(nodes[1:])
        while remaining:
            best = float('inf')
            best_node = None
            for u in in_mst:
                for v in remaining:
                    if dist_matrix[u][v] < best:
                        best = dist_matrix[u][v]
                        best_node = v
            cost_val += best
            in_mst.add(best_node)
            remaining.remove(best_node)
        return cost_val

    def heuristic(current: int, visited: int) -> float:
        """Heuristic: minimal distance from current to any unvisited node plus the MST cost on the unvisited nodes."""
        unvisited_mask = ((1 << n) - 1) ^ visited
        if unvisited_mask == 0:
            return 0.0
        h1 = min(dist_matrix[current][u] for u in range(n) if unvisited_mask & (1 << u))
        h2 = mst_cost(unvisited_mask)
        return h1 + h2

    def tsp_astar_cover_bitmask(start_index: int) -> t.Tuple[t.List[int], float]:
        initial_mask = 1 << start_index
        # The state is a tuple: (f = cost+heuristic, cost_so_far, current_index, visited_mask, route)
        initial_state = (heuristic(start_index, initial_mask), 0.0, start_index, initial_mask, [start_index])
        frontier = []
        heapq.heappush(frontier, initial_state)
        best_cost_state: t.Dict[t.Tuple[int, int], float] = {}
        goal_mask = (1 << n) - 1
        iterations = 0

        while frontier:
            iterations += 1
            f, cost, current, visited, route = heapq.heappop(frontier)
            if visited == goal_mask:
                return route, cost

            state_key = (current, visited)
            if state_key in best_cost_state and best_cost_state[state_key] <= cost:
                continue
            best_cost_state[state_key] = cost

            # Build neighbor list:
            #   Always include unvisited nodes.
            unvisited = [u for u in range(n) if (visited & (1 << u)) == 0 and u != current]
            #   Also consider some already–visited nodes as potential shortcuts.
            visited_candidates = [u for u in range(n) if (visited & (1 << u)) and u != current]
            # Sort both lists by the travel cost from current.
            unvisited_sorted = sorted(unvisited, key=lambda u: dist_matrix[current][u])
            visited_sorted = sorted(visited_candidates, key=lambda u: dist_matrix[current][u])
            # If there are many unvisited nodes, limit them;
            # if not, add a few visited nodes as well.
            if len(unvisited_sorted) >= neighbor_limit:
                neighbors = unvisited_sorted[:neighbor_limit]
            else:
                additional = visited_sorted[: max(neighbor_limit - len(unvisited_sorted), 0)]
                neighbors = unvisited_sorted + additional

            # Expand each neighbor.
            for nxt in neighbors:
                d = dist_matrix[current][nxt]
                if d == float('inf'):
                    continue
                new_cost = cost + d
                # If nxt has not been visited, update the mask.
                if visited & (1 << nxt):
                    new_visited = visited
                else:
                    new_visited = visited | (1 << nxt)
                new_route = route + [nxt]
                h = heuristic(nxt, new_visited)
                heapq.heappush(frontier, (new_cost + h, new_cost, nxt, new_visited, new_route))

            # Beam search: prune the frontier every 100 iterations.
            if iterations % 100 == 0 and len(frontier) > beam_width:
                frontier = heapq.nsmallest(beam_width, frontier)
                heapq.heapify(frontier)
        return None, float('inf')

    start_index = mapping[start_station]
    route_indices, total_cost = tsp_astar_cover_bitmask(start_index)
    route_station_ids = [inv_mapping[i] for i in route_indices]
    return start_station, route_station_ids, total_cost


def fast_worker_wrapper(start_station: str,
                        mapping: t.Dict[str, int],
                        inv_mapping: t.Dict[int, str],
                        n: int,
                        dist_matrix: t.List[t.List[float]]
                        ) -> t.Tuple[str, t.List[str], float]:
    return fast_worker_astar_cover_with_revisits(start_station, mapping, inv_mapping, n, dist_matrix)


# -----------------------
# Main routine
# -----------------------
def main():
    lines, stations, line_routes = build_network()
    G = build_graph(stations, raw_intervals)

    station_list, mapping, inv_mapping, n, dist_matrix = prepare_search(G)

    args = [(station, mapping, inv_mapping, n, dist_matrix) for station in station_list]

    with multiprocessing.Pool() as pool:
        results = pool.starmap(fast_worker_wrapper, args)

    processed_results = [
        {
            'start_station': start_station,
            'route': route,
            'total_time': total_cost
        }
        for start_station, route, total_cost in results
    ]

    with open('results.json', 'w') as f:
        json.dump(processed_results, f, indent=4)


if __name__ == '__main__':
    main()
