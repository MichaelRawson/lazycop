use lazy::prelude::*;

pub struct RuleList {
    parent: Option<Id<RuleList>>,
    rule: Rule,
    heuristic: u16,
    actual: u16,
}

#[derive(Default)]
pub struct RuleStore {
    tree: Block<RuleList>,
}

impl RuleStore {
    pub fn get_actual(&self, id: Id<RuleList>) -> u16 {
        self.tree[id].actual
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
        let actual = std::u16::MAX;
        let list = RuleList {
            parent,
            rule,
            heuristic,
            actual
        };
        self.tree.push(list)
    }

    pub fn recompute_heuristics(&mut self) {
        for id in self.tree.range().rev() {
            let parent = if let Some(parent) = self.tree[id].parent {
                parent
            }
            else {
                continue;
            };
            let actual = self.get_actual(id);
            let actual = if actual == std::u16::MAX {
                self.tree[id].heuristic
            }
            else {
                actual
            };
            let distance = actual + 1;
            let parent_distance = self.tree[parent].actual;
            let recomputed = std::cmp::min(distance, parent_distance);
            self.tree[parent].actual = recomputed;
        }
    }

    pub fn examples(&self) -> impl Iterator<Item = Id<RuleList>> + '_ {
        self.tree
            .range()
            .rev()
            .filter(move |id| self.tree[*id].actual != std::u16::MAX)
    }
}
