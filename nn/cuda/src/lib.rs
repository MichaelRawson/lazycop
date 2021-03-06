pub struct Input<'a> {
    pub nodes: &'a [u32],
    pub sources: &'a [u32],
    pub targets: &'a [u32],
    pub rules: &'a [u32],
}

mod cuda {
    #[link(name = "model")]
    extern "C" {
        pub fn init();
        pub fn model(
            num_nodes: u32,
            num_edges: u32,
            num_rules: u32,
            nodes: *const u32,
            sources: *const u32,
            targets: *const u32,
            rules: *const u32,
            buf: *mut f32,
        );
    }
}

pub fn init() {
    unsafe { cuda::init() };
}

pub fn model(input: Input, output: &mut Vec<f32>) {
    output.resize_with(input.rules.len(), Default::default);

    debug_assert!(input.rules.len() > 1);
    debug_assert!(input.sources.len() == input.targets.len());
    let num_nodes = input.nodes.len() as u32;
    let num_edges = input.sources.len() as u32;
    let num_rules = input.rules.len() as u32;

    let nodes = input.nodes.as_ptr();
    let rules = input.rules.as_ptr();
    let sources = input.sources.as_ptr();
    let targets = input.targets.as_ptr();

    let buf = output.as_mut_ptr();
    unsafe {
        cuda::model(
            num_nodes, num_edges, num_rules, nodes, sources, targets, rules,
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
