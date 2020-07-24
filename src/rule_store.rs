use crate::prelude::*;

pub(crate) struct RuleList {
    parent: Option<Id<RuleList>>,
    count: u32,
    rule: Rule,
}

#[derive(Default)]
pub(crate) struct RuleStore {
    tree: Block<RuleList>,
    free: Vec<Id<RuleList>>,
}

impl RuleStore {
    pub(crate) fn get_list(
        &self,
        leaf: Id<RuleList>,
    ) -> impl Iterator<Item = Rule> + '_ {
        let mut current = Some(leaf);
        std::iter::from_fn(move || {
            let list = &self.tree[current?];
            let rule = Some(list.rule);
            current = list.parent;
            rule
        })
    }

    pub(crate) fn add(
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

    pub(crate) fn mark_done(&mut self, leaf: Id<RuleList>) -> u16 {
        let mut closed = 0;
        let mut current = Some(leaf);
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