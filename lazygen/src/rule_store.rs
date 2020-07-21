use lazy::prelude::*;

pub struct RuleList {
    parent: Option<Id<RuleList>>,
    rule: Rule,
    expanded: bool,
    heuristic: u16,
    estimate: u16,
}

#[derive(Default)]
pub struct RuleStore {
    tree: Block<RuleList>,
}

impl RuleStore {
    pub fn get_heuristic(&self, id: Id<RuleList>) -> u16 {
        self.tree[id].heuristic
    }

    pub fn get_estimate(&self, id: Id<RuleList>) -> u16 {
        self.tree[id].estimate
    }

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
        heuristic: u16,
    ) -> Id<RuleList> {
        let expanded = false;
        let heuristic = heuristic as u16;
        let estimate = heuristic;
        let list = RuleList {
            parent,
            rule,
            expanded,
            heuristic,
            estimate,
        };
        self.tree.push(list)
    }

    pub fn mark_expanded(&mut self, id: Id<RuleList>) {
        self.tree[id].expanded = true;
        self.tree[id].estimate = std::u16::MAX;
    }

    pub fn bubble_up(&mut self) {
        for id in self.tree.range().rev() {
            let estimate = self.tree[id].heuristic;
            let parent = if let Some(parent) = self.tree[id].parent {
                parent
            } else {
                continue;
            };

            let parent_estimate = 1 + estimate;
            if parent_estimate < self.tree[parent].estimate {
                self.tree[parent].estimate = parent_estimate;
            }
        }
    }

    pub fn examples(&self) -> impl Iterator<Item = Id<RuleList>> + '_ {
        self.tree
            .range()
            .filter(move |id| self.tree[*id].expanded)
            .filter(move |id| self.tree[*id].estimate != std::u16::MAX)
    }
}
