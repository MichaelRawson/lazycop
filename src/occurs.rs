pub trait Occurs {
    const CHECK: bool;
}

pub struct Check;

impl Occurs for Check {
    const CHECK: bool = true;
}

pub struct SkipCheck;

impl Occurs for SkipCheck {
    const CHECK: bool = false;
}
