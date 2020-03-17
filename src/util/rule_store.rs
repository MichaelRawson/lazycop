use crate::prelude::*;

#[derive(Default)]
pub struct RuleStore {
    rules: Vec<(Id<Rule>, Rule)>
}

impl RuleStore {
    pub fn add_rule(&mut self, parent: Id<Rule>, rule: Rule) -> Id<Rule> {
        let id = self.rules.len().into();
        self.rules.push((parent, rule));
        id
    }

    pub fn add_start_rule(&mut self, rule: Rule) -> Id<Rule> {
        let parent = self.rules.len().into();
        self.add_rule(parent, rule)
    }

    pub fn get_script(&self, end: Id<Rule>) -> Vec<Rule> {
        let mut current = end;
        let mut script = vec![];
        loop {
            let (next, rule) = self.rules[current.index()];
            script.push(rule);
            if current == next {
                break;
            }
            current = next;
        }
        script.reverse();
        script
    }
}
