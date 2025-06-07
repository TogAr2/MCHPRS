//! # [`PruneOrphans`]
//!
//! This pass removes any nodes in the graph that aren't transitively connected to an output redstone component by using Depth-First-Search.

use super::Pass;
use crate::compile_graph::{CompileGraph, NodeIdx};
use crate::{CompilerInput, CompilerOptions};
use itertools::Itertools;
use mchprs_world::{TickEntry, World};
use petgraph::Direction;
use rustc_hash::{FxHashMap, FxHashSet};

pub struct PruneOrphans;

impl<W: World> Pass<W> for PruneOrphans {
    fn run_pass(
        &self,
        graph: &mut CompileGraph,
        _options: &CompilerOptions,
        _ticks: &mut Vec<TickEntry>,
        _link_breaks: &mut FxHashMap<NodeIdx, usize>,
        _input: &CompilerInput<'_, W>,
    ) {
        let mut to_visit = graph
            .node_indices()
            .filter(|&idx| !graph[idx].is_removable())
            .collect_vec();

        let mut visited = FxHashSet::default();
        while let Some(idx) = to_visit.pop() {
            if visited.insert(idx) {
                to_visit.extend(graph.neighbors_directed(idx, Direction::Incoming));
            }
        }

        graph.retain_nodes(|_, idx| visited.contains(&idx));
    }

    fn should_run(&self, options: &CompilerOptions) -> bool {
        options.io_only && options.optimize
    }

    fn status_message(&self) -> &'static str {
        "Pruning orphans"
    }
}
