__all__ = ['TidemanRankedPairsAggregator']

import numpy as np
from .base import RankAggregator
from .utils import ranks_from_preferences


class TidemanRankedPairsAggregator(RankAggregator):
    """
    Tideman's Ranked Pairs aggregator.
    
    Tideman's method is a Condorcet method that:
    1. Computes pairwise comparison margins for all candidate pairs
    2. Sorts pairs by margin (largest margins first)
    3. Locks in pairs one by one, but only if they don't create a cycle
    4. Builds the final ranking from the locked pairs
    
    This method satisfies the Condorcet criterion: if a candidate beats all others
    in pairwise comparisons, that candidate wins.
    
    Reference: https://en.wikipedia.org/wiki/Ranked_pairs
    """
    
    def aggregate(self, preferences: np.ndarray) -> np.ndarray:
        """
        Aggregate preferences using Tideman's ranked pairs method.
        
        Args:
            preferences: m x n preference matrix where each row is a ranking.
                        Values are item indices in ranked order.
        
        Returns:
            Aggregated preference as 1D array of length n.
        """
        m, n = preferences.shape
        
        # Convert preferences to ranks
        ranks = ranks_from_preferences(preferences)  # m x n
        
        # Compute pairwise comparison matrix
        # margin[i, j] = number of times i beats j - number of times j beats i
        # Positive value means i beats j, negative means j beats i
        margin_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                # Count how many times i is ranked before j
                i_before_j = np.sum((ranks[:, i] < ranks[:, j]) & 
                                   (ranks[:, i] != -1) & (ranks[:, j] != -1))
                # Count how many times j is ranked before i
                j_before_i = np.sum((ranks[:, j] < ranks[:, i]) & 
                                   (ranks[:, i] != -1) & (ranks[:, j] != -1))
                
                margin = i_before_j - j_before_i
                margin_matrix[i, j] = margin
                margin_matrix[j, i] = -margin
        
        # Get all pairs with their margins, sorted by absolute margin (descending)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                margin = margin_matrix[i, j]
                if margin > 0:
                    pairs.append((i, j, margin))
                elif margin < 0:
                    pairs.append((j, i, -margin))
                # If margin == 0, we skip the pair (tie)
        
        # Sort pairs by margin (largest first)
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Build locked graph by adding pairs one by one, avoiding cycles
        locked_edges = set()
        locked_graph = np.zeros((n, n), dtype=bool)  # Adjacency matrix for locked edges
        
        for winner, loser, margin in pairs:
            # Check if adding this edge would create a cycle
            if not self._would_create_cycle(locked_graph, winner, loser):
                locked_edges.add((winner, loser))
                locked_graph[winner, loser] = True
        
        # Build ranking from locked graph using topological sort
        ranking = self._topological_sort(locked_graph)
        
        return ranking
    
    def _would_create_cycle(self, graph: np.ndarray, from_node: int, to_node: int) -> bool:
        """
        Check if adding an edge from from_node to to_node would create a cycle.
        
        Uses DFS to check if there's already a path from to_node to from_node.
        If such a path exists, adding the edge would create a cycle.
        
        Args:
            graph: Current adjacency matrix of locked edges
            from_node: Source node of the edge to add
            to_node: Destination node of the edge to add
        
        Returns:
            True if adding the edge would create a cycle, False otherwise
        """
        # If there's a path from to_node to from_node, adding from_node -> to_node creates a cycle
        visited = np.zeros(graph.shape[0], dtype=bool)
        return self._has_path(graph, to_node, from_node, visited)
    
    def _has_path(self, graph: np.ndarray, start: int, end: int, visited: np.ndarray) -> bool:
        """
        Check if there's a path from start to end using DFS.
        
        Args:
            graph: Adjacency matrix
            start: Starting node
            end: Target node
            visited: Visited array to avoid revisiting nodes
        
        Returns:
            True if path exists, False otherwise
        """
        if start == end:
            return True
        
        visited[start] = True
        
        for neighbor in range(graph.shape[0]):
            if graph[start, neighbor] and not visited[neighbor]:
                if self._has_path(graph, neighbor, end, visited):
                    return True
        
        return False
    
    def _topological_sort(self, graph: np.ndarray) -> np.ndarray:
        """
        Perform topological sort on the locked graph to get the final ranking.
        
        Args:
            graph: Adjacency matrix of locked edges
        
        Returns:
            Array of node indices in topological order (best to worst)
        """
        n = graph.shape[0]
        in_degree = graph.sum(axis=0)  # Count incoming edges for each node
        
        # Find nodes with no incoming edges (sources)
        queue = [i for i in range(n) if in_degree[i] == 0]
        result = []
        
        while queue:
            # Process nodes with no incoming edges
            node = queue.pop(0)
            result.append(node)
            
            # Remove this node and update in-degrees
            for neighbor in range(n):
                if graph[node, neighbor]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
        
        # If we didn't process all nodes, there might be cycles (shouldn't happen with ranked pairs)
        # But handle it gracefully by adding remaining nodes
        remaining = [i for i in range(n) if i not in result]
        result.extend(remaining)
        
        return np.array(result)

