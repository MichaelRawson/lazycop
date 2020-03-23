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

    pub fn rules(&self) -> Vec<Rule> {
        let mut rules = vec![self.rule];
        let mut current = self.parent.as_ref();
        while let Some(next) = current {
            rules.push(next.rule);
            current = next.parent.as_ref();
        }
        rules.reverse();
        rules
    }
}
