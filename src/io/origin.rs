use std::path::PathBuf;
use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct Origin {
    pub(crate) conjecture: bool,
    pub(crate) path: Arc<PathBuf>,
    pub(crate) name: Arc<String>,
}
