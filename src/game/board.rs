#![allow(dead_code)]

use std::collections::HashSet;

const MAX_NEIGHBORS: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VertexId(usize);

impl VertexId {
    pub fn new(index: usize) -> Self {
        Self(index)
    }

    pub fn index(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoardError {
    UnknownVertex(VertexId),
    SelfLoop(VertexId),
    TooManyNeighbors(VertexId),
    DuplicateEdge(VertexId, VertexId),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoardGraph {
    neighbors: Vec<Vec<VertexId>>,
    max_degree: usize,
}

impl BoardGraph {
    pub fn from_edges(
        vertex_count: usize,
        edges: impl IntoIterator<Item = (VertexId, VertexId)>,
    ) -> Result<Self, BoardError> {
        let mut adjacency = vec![Vec::new(); vertex_count];
        let mut seen = HashSet::new();

        for (left, right) in edges {
            if left.index() >= vertex_count {
                return Err(BoardError::UnknownVertex(left));
            }
            if right.index() >= vertex_count {
                return Err(BoardError::UnknownVertex(right));
            }
            if left == right {
                return Err(BoardError::SelfLoop(left));
            }

            let edge = if left < right {
                (left, right)
            } else {
                (right, left)
            };

            if !seen.insert(edge) {
                return Err(BoardError::DuplicateEdge(edge.0, edge.1));
            }

            adjacency[left.index()].push(right);
            adjacency[right.index()].push(left);
        }
        let mut max_degree = 0;
        for (index, neighbors) in adjacency.iter_mut().enumerate() {
            if neighbors.len() > MAX_NEIGHBORS {
                return Err(BoardError::TooManyNeighbors(VertexId::new(index)));
            }
            if neighbors.len() > max_degree {
                max_degree = neighbors.len();
            }
            neighbors.sort_unstable();
        }

        Ok(Self {
            neighbors: adjacency,
            max_degree,
        })
    }

    /// Returns the symmetrically normalized adjacency matrix with self-loops added, shape [N, N].
    pub fn normalized_adjacency(&self) -> Vec<Vec<f32>> {
        let n = self.neighbors.len();
        let mut adj = vec![vec![0.0f32; n]; n];

        for (i, row) in adj.iter_mut().enumerate() {
            for &j in &self.neighbors[i] {
                row[j.index()] = 1.0;
            }
        }

        for (i, row) in adj.iter_mut().enumerate() {
            row[i] = 1.0;
        }

        let deg: Vec<f32> = adj.iter().map(|row| row.iter().sum()).collect();

        let d_inv_sqrt: Vec<f32> = deg
            .iter()
            .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 })
            .collect();

        for (i, row) in adj.iter_mut().enumerate() {
            for (j, val) in row.iter_mut().enumerate() {
                *val = d_inv_sqrt[i] * *val * d_inv_sqrt[j];
            }
        }

        adj
    }

    pub fn max_degree(&self) -> usize {
        self.max_degree
    }

    pub fn vertex_count(&self) -> usize {
        self.neighbors.len()
    }

    pub fn contains(&self, vertex: VertexId) -> bool {
        vertex.index() < self.vertex_count()
    }

    pub fn get_neighbors(&self, vertex: VertexId) -> Option<&[VertexId]> {
        self.neighbors.get(vertex.index()).map(Vec::as_slice)
    }

    pub fn are_adjacent(&self, left: VertexId, right: VertexId) -> bool {
        self.get_neighbors(left)
            .is_some_and(|neighbors| neighbors.contains(&right))
    }

    pub fn vertices(&self) -> impl ExactSizeIterator<Item = VertexId> {
        (0..self.vertex_count()).map(VertexId::new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vertex_store() {
        let vertex = VertexId::new(42);
        assert_eq!(vertex.index(), 42)
    }

    #[test]
    fn test_vertex_equal() {
        let v1 = VertexId::new(42);
        let v2 = VertexId::new(42);
        let v3 = VertexId::new(9);
        assert_eq!(v1, v2);
        assert_ne!(v3, v1);
    }

    #[test]
    fn test_board_create() {
        assert_eq!(
            BoardGraph::from_edges(1, vec![(VertexId::new(0), VertexId::new(1))]),
            Err(BoardError::UnknownVertex(VertexId::new(1)))
        );

        assert_eq!(
            BoardGraph::from_edges(
                2,
                vec![
                    (VertexId::new(0), VertexId::new(1)),
                    (VertexId::new(1), VertexId::new(0))
                ]
            ),
            Err(BoardError::DuplicateEdge(VertexId::new(0), VertexId(1)))
        );

        assert_eq!(
            BoardGraph::from_edges(
                2,
                vec![
                    (VertexId::new(0), VertexId::new(0)),
                    (VertexId::new(0), VertexId::new(1))
                ]
            ),
            Err(BoardError::SelfLoop(VertexId::new(0)))
        );

        assert_eq!(
            BoardGraph::from_edges(
                5,
                vec![
                    (VertexId::new(0), VertexId::new(1)),
                    (VertexId::new(0), VertexId::new(2)),
                    (VertexId::new(0), VertexId::new(3)),
                    (VertexId::new(0), VertexId::new(4)),
                ]
            ),
            Err(BoardError::TooManyNeighbors(VertexId(0)))
        );

        assert!(
            BoardGraph::from_edges(
                5,
                vec![
                    (VertexId::new(0), VertexId::new(1)),
                    (VertexId::new(0), VertexId::new(2)),
                    (VertexId::new(0), VertexId::new(3)),
                    (VertexId::new(1), VertexId::new(4)),
                    (VertexId::new(3), VertexId::new(2)),
                ]
            )
            .is_ok()
        );
    }

    #[test]
    fn test_board_function() {
        let board = BoardGraph::from_edges(
            5,
            vec![
                (VertexId::new(0), VertexId::new(1)),
                (VertexId::new(0), VertexId::new(2)),
                (VertexId::new(0), VertexId::new(3)),
                (VertexId::new(1), VertexId::new(4)),
                (VertexId::new(3), VertexId::new(2)),
            ],
        )
        .unwrap();

        assert_eq!(board.vertex_count(), 5);
        assert!(board.contains(VertexId::new(3)));
        assert!(!board.contains(VertexId::new(5)));

        assert!(board.get_neighbors(VertexId::new(0)).is_some_and(
            |neighbors| neighbors == [VertexId::new(1), VertexId::new(2), VertexId::new(3)]
        ));

        assert!(board.are_adjacent(VertexId::new(4), VertexId::new(1)));

        assert!(!board.are_adjacent(VertexId::new(4), VertexId::new(2)));

        for (i, vertex) in board.vertices().enumerate() {
            assert_eq!(VertexId::new(i), vertex);
        }
    }
}
