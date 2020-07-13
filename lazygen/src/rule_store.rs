use lazy::prelude::*;

pub struct RuleList {
    parent: Option<Id<RuleList>>,
    rule: Rule,
    expanded: bool,
    heuristic: u16,
}

#[derive(Default)]
pub struct RuleStore {
    tree: Block<RuleList>,
}

impl RuleStore {
    pub fn get_heuristic(&self, id: Id<RuleList>) -> u16 {
        self.tree[id].heuristic
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
        let list = RuleList {
            parent,
            rule,
            expanded,
            heuristic,
        };
        self.tree.push(list)
    }

    pub fn mark_expanded(&mut self, id: Id<RuleList>) {
        self.tree[id].expanded = true;
    }

    pub fn recompute_heuristics(&mut self) {
        for id in self.tree.range() {
            if self.tree[id].expanded {
                self.tree[id].heuristic = std::u16::MAX;
            }
        }

        for id in self.tree.range().rev() {
            let heuristic = self.tree[id].heuristic;
            if heuristic == std::u16::MAX {
                continue;
            }
            let parent = if let Some(parent) = self.tree[id].parent {
                parent
            } else {
                continue;
            };
            let distance = heuristic + 1;
            let parent_distance = self.tree[parent].heuristic;
            let recomputed = std::cmp::min(distance, parent_distance);
            self.tree[parent].heuristic = recomputed;
        }
    }

    pub fn examples(&self) -> impl Iterator<Item = Id<RuleList>> + '_ {
        self.tree
            .range()
            .rev()
            .filter(move |id| self.tree[*id].expanded)
            .filter(move |id| self.tree[*id].heuristic != std::u16::MAX)
    }
}
