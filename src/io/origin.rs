use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct Origin {
    pub(crate) fof: bool,
    pub(crate) conjecture: bool,
    pub(crate) path: Arc<String>,
    pub(crate) name: Arc<String>,
}
