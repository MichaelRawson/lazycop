use crate::prelude::*;

pub struct Script {
    parent: Option<Rc<Script>>,
    rule: Rule,
}

impl Script {
    pub fn new(parent: Rc<Self>, rule: Rule) -> Rc<Self> {
        let parent = Some(parent);
        Rc::new(Self { parent, rule })
    }

    pub fn start(rule: Rule) -> Rc<Self> {
        let parent = None;
        Rc::new(Self { rule, parent })
    }

    pub fn fill_rules(&self, rules: &mut Vec<Rule>) {
        rules.clear();
        rules.push(self.rule);
        let mut current = self.parent.as_ref();
        while let Some(next) = current {
            rules.push(next.rule);
            current = next.parent.as_ref();
        }
        rules.reverse()
    }
}
