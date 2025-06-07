use rustc_hash::FxHashMap;
use super::Pass;
use crate::compile_graph::{CompileGraph, NodeIdx};
use crate::{CompilerInput, CompilerOptions};
use mchprs_world::{TickEntry, World};

pub struct ClampWeights;

impl<W: World> Pass<W> for ClampWeights {
    fn run_pass(
        &self,
        graph: &mut CompileGraph,
        _options: &CompilerOptions,
        _ticks: &mut Vec<TickEntry>,
        _link_breaks: &mut FxHashMap<NodeIdx, usize>,
        _input: &CompilerInput<'_, W>
    ) {
        graph.retain_edges(|g, edge| g[edge].ss < 15);
    }

    fn should_run(&self, _: &CompilerOptions) -> bool {
        // Mandatory
        true
    }

    fn status_message(&self) -> &'static str {
        "Clamping weights"
    }
}
