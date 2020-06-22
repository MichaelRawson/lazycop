use crate::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Start {
    pub clause: Id<ProblemClause>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PredicateReduction {
    pub literal: Id<Literal>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PredicateExtension {
    pub occurrence: Id<PredicateOccurrence>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct EqualityReduction {
    pub literal: Id<Literal>,
    pub target: Id<Term>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct EqualityExtension {
    pub target: Id<Term>,
    pub occurrence: Id<EqualityOccurrence>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SubtermExtension {
    pub occurrence: Id<SubtermOccurrence>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Rule {
    Start(Start),
    Reflexivity,
    PredicateReduction(PredicateReduction),
    LREqualityReduction(EqualityReduction),
    RLEqualityReduction(EqualityReduction),
    LRSubtermReduction(EqualityReduction),
    RLSubtermReduction(EqualityReduction),
    StrictPredicateExtension(PredicateExtension),
    LazyPredicateExtension(PredicateExtension),
    StrictFunctionExtension(EqualityExtension),
    LazyFunctionExtension(EqualityExtension),
    VariableExtension(EqualityExtension),
    LRLazySubtermExtension(SubtermExtension),
    RLLazySubtermExtension(SubtermExtension),
    LRStrictSubtermExtension(SubtermExtension),
    RLStrictSubtermExtension(SubtermExtension),
}

impl Rule {
    pub fn lr(&self) -> bool {
        match self {
            Rule::LREqualityReduction(_)
            | Rule::LRSubtermReduction(_)
            | Rule::LRLazySubtermExtension(_)
            | Rule::LRStrictSubtermExtension(_) => true,
            Rule::RLEqualityReduction(_)
            | Rule::RLSubtermReduction(_)
            | Rule::RLLazySubtermExtension(_)
            | Rule::RLStrictSubtermExtension(_) => false,
            _ => unreachable(),
        }
    }

    pub fn precedence(&self) -> u16 {
        match self {
            Rule::Start(_) => 0,
            Rule::Reflexivity | Rule::PredicateReduction(_) => 1,
            Rule::LREqualityReduction(_)
            | Rule::RLEqualityReduction(_)
            | Rule::LRSubtermReduction(_)
            | Rule::RLSubtermReduction(_) => 2,
            Rule::StrictPredicateExtension(_)
            | Rule::StrictFunctionExtension(_) => 3,
            Rule::LazyPredicateExtension(_)
            | Rule::LazyFunctionExtension(_) => 4,
            Rule::VariableExtension(_) => 5,
            Rule::LRStrictSubtermExtension(_)
            | Rule::RLStrictSubtermExtension(_) => 6,
            Rule::LRLazySubtermExtension(_)
            | Rule::RLLazySubtermExtension(_) => 7,
        }
    }
}

pub struct RuleList {
    parent: Option<Id<RuleList>>,
    count: u32,
    rule: Rule,
}

#[derive(Default)]
pub struct Rules {
    tree: Block<RuleList>,
    free: Vec<Id<RuleList>>,
}

impl Rules {
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
