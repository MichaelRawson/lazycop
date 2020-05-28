pub(crate) trait Occurs {
    const CHECK: bool;
}

pub(crate) struct Check;

impl Occurs for Check {
    const CHECK: bool = true;
}

pub(crate) struct SkipCheck;

impl Occurs for SkipCheck {
    const CHECK: bool = false;
}
