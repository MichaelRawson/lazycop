pub struct Input<'a> {
    pub num_graphs: u32,
    pub nodes: &'a [u32],
    pub sources: &'a [u32],
    pub targets: &'a [u32],
    pub batch: &'a [u32],
}

mod cuda {
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
            buf: *mut f32,
        );
    }
}

pub fn init() {
    unsafe { cuda::init() };
}

pub fn model(input: Input, output: &mut Vec<f32>) {
    output.resize_with(input.num_graphs as usize, Default::default);
    if input.num_graphs == 1 {
        output[0] = 1.0;
        return;
    }

    debug_assert!(input.num_graphs > 0);
    debug_assert!(input.sources.len() == input.targets.len());
    let num_nodes = input.nodes.len() as u32;
    let num_edges = input.sources.len() as u32;
    let num_graphs = input.num_graphs;

    let nodes = input.nodes.as_ptr();
    let batch = input.batch.as_ptr();
    let sources = input.sources.as_ptr();
    let targets = input.targets.as_ptr();

    let buf = output.as_mut_ptr();
    unsafe {
        cuda::model(
            num_nodes, num_edges, num_graphs, nodes, sources, targets, batch,
            buf,
        )
    };

    let max_logit = output
        .iter()
        .max_by(|x, y| x.partial_cmp(y).expect("bad float comparison"))
        .copied()
        .unwrap_or(0.0);
    let log_sum_exp = output
        .iter()
        .map(|logit| (logit - max_logit).exp())
        .sum::<f32>()
        .ln();
    for logit in output {
        *logit -= log_sum_exp;
        *logit -= max_logit;
    }
}
