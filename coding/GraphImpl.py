import igraph
import pandas as pd


class Jgraph(object):
    def __init__(self, nodes, edges, directed=False):
        self.g = igraph.Graph(directed=directed)

        for node in nodes:
            self.g.add_vertex(node['name'], weight=node.get('value'))
        for edge in edges:
            self.g.add_edge(edge['source'], edge['target'], weight=edge.get('value'))

        self.N = len(self.g.vs)

        self.edge_weights = None if None in self.g.es['weight'] else self.g.es['weight']

    def Series(self, keys, values, name):
        series = pd.Series(dict(zip(keys, values)))
        series.name = name
        return series

    def PageRankSolution(self):
        pass

    def shortest_paths_Solve(self, srcNodes, tgtNodes, mode="single", show=True):
        """
        最短路径，单源点最短路径，还是多源点间最短路径
        :param srcNodes: 源节点
        :param tgtNodes: 目标节点（可能是集合）
        :param mode: single/multiple
        :param show:
        :return:
        """
        assert mode.lower() in ['single', 'multiple'], "不支持的模式"

        if mode == "multiple":
            searched_shortest_paths = []
            for single_source in srcNodes:
                for single_target in tgtNodes:
                    searched_shortest_paths.append(
                        self.__search_shortest_paths(single_source, single_target, show))
            return searched_shortest_paths
        else:
            # 单源点单终点
            return self.__search_shortest_paths(srcNodes,tgtNodes,show)

    def __search_shortest_paths(self, single_source, single_target, show=True):
        try:
            shortest_paths_ids = self.g.get_all_shortest_paths(single_source, single_target, weights=self.edge_weights)
            shortest_paths_names = [
                [self.g.vs[id]['name'] for id in shortest_path_id]
                for shortest_path_id in shortest_paths_ids
            ]

            if self.edge_weights:
                shortest_paths_lengths = [
                    sum(self.g.es[
                        self.g.get_eids(path=shortest_paths_ids[i])]['weight']) for i in range(len(shortest_paths_ids))]
            else:
                shortest_paths_lengths = [len(shortest_paths_ids[i]) - 1 for i in range(len(shortest_paths_ids))]
            if show:
                for (d, shortest_path_name) in enumerate(shortest_paths_names):
                    print(f"Path {d + 1} ({shortest_paths_lengths[d]}): {' -> '.join(shortest_path_name)}")
            return list(zip(shortest_paths_names, shortest_paths_lengths))

        except IndexError:
            return [[], float("inf")]


