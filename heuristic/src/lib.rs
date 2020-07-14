use lazy::prelude::*;

mod model {
    #[link(name = "model")]
    extern "C" {
        pub fn init();
        pub fn model(
            num_nodes: u32,
            num_edges: u32,
            num_graphs: u32,
            nodes: *const u32,
            sources: *const u32,
            targets: *const u32,
            batch: *const u32,
            results: *mut f32,
        );
    }
}

pub fn init() {
    unsafe { model::init() };
}

pub fn model(batch: &Graph, result_buf: &mut [f32]) {
    let num_nodes = batch.nodes.len().as_u32() - 1;
    let num_edges = batch.from.len() as u32;
    let num_graphs = batch.subgraphs;
    let nodes = batch.nodes.as_slice().as_ptr() as *const u32;
    let sources = batch.from.as_ptr();
    let targets = batch.to.as_ptr();
    let batch = batch.batch.as_ptr();
    let results = result_buf.as_mut_ptr();
    if num_graphs == 0 {
        return;
    }
    unsafe {
        model::model(
            num_nodes, num_edges, num_graphs, nodes, sources, targets, batch,
            results,
        )
    };
}
