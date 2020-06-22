use lazy::prelude::*;

pub struct RuleList {
    parent: Option<Id<RuleList>>,
    count: u32,
    rule: Rule,
}

#[derive(Default)]
pub struct RuleStore {
    tree: Block<RuleList>,
    free: Vec<Id<RuleList>>,
}

impl RuleStore {
    pub fn get_list(
        &self,
        mut current: Option<Id<RuleList>>,
    ) -> impl Iterator<Item = Rule> + '_ {
        std::iter::from_fn(move || {
            let list = &self.tree[current?];
            let rule = Some(list.rule);
            current = list.parent;
            rule
        })
    }

    pub fn add(
        &mut self,
        parent: Option<Id<RuleList>>,
        rule: Rule,
    ) -> Id<RuleList> {
        if let Some(parent) = parent {
            self.tree[parent].count += 1;
        }

        let count = 1;
        let list = RuleList {
            parent,
            count,
            rule,
        };
        if let Some(free) = self.free.pop() {
            self.tree[free] = list;
            free
        } else {
            self.tree.push(list)
        }
    }

    pub fn mark_done(&mut self, mut current: Option<Id<RuleList>>) -> u16 {
        let mut closed = 0;
        while let Some(id) = current {
            let leaf = &mut self.tree[id];
            leaf.count -= 1;
            if leaf.count > 0 {
                break;
            }
            self.free.push(id);
            current = leaf.parent;
            closed += 1;
        }
        closed
    }
}
