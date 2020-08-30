use std::rc::Rc;

pub(crate) struct Origin {
    pub(crate) conjecture: bool,
    pub(crate) path: Rc<String>,
    pub(crate) name: String,
}
