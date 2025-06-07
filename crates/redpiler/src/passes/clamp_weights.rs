use super::Pass;
use crate::compile_graph::CompileGraph;
use crate::{CompilerInput, CompilerOptions, RuntimeAction};
use mchprs_world::World;

pub struct ClampWeights;

impl<W: World> Pass<W> for ClampWeights {
    fn run_pass(
        &self,
        graph: &mut CompileGraph,
        options: &CompilerOptions,
        actions: &mut Vec<RuntimeAction>,
        input: &CompilerInput<'_, W>
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
