use crate::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Start {
    pub(crate) clause: Id<ProblemClause>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct PredicateReduction {
    pub(crate) literal: Id<Literal>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct PredicateExtension {
    pub(crate) occurrence: Id<PredicateOccurrence>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct EqualityReduction {
    pub(crate) literal: Id<Literal>,
    pub(crate) target: Id<Term>,
    pub(crate) from: Id<Term>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct EqualityExtension {
    pub(crate) target: Id<Term>,
    pub(crate) occurrence: Id<EqualityOccurrence>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Rule {
    Start(Start),
    Reflexivity,
    PredicateReduction(PredicateReduction),
    EqualityReduction(EqualityReduction),
    StrictPredicateExtension(PredicateExtension),
    LazyPredicateExtension(PredicateExtension),
    StrictFunctionExtension(EqualityExtension),
    LazyFunctionExtension(EqualityExtension),
    VariableExtension(EqualityExtension),
}

impl Rule {
    pub(crate) fn precedence(&self) -> u16 {
        match self {
            Rule::Start(_) => 0,
            Rule::Reflexivity => 1,
            Rule::PredicateReduction(_) => 1,
            Rule::EqualityReduction(_) => 2,
            Rule::StrictPredicateExtension(_) => 3,
            Rule::StrictFunctionExtension(_) => 3,
            Rule::LazyPredicateExtension(_) => 4,
            Rule::LazyFunctionExtension(_) => 4,
            Rule::VariableExtension(_) => 5,
        }
    }
}

pub(crate) struct RuleList {
    parent: Option<Id<RuleList>>,
    count: u32,
    rule: Rule,
}

#[derive(Default)]
pub(crate) struct Rules {
    tree: Block<RuleList>,
    free: Vec<Id<RuleList>>,
}

impl Rules {
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
        let count = 0;
        if let Some(parent) = parent {
            self.tree[parent].count += 1;
        }

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

    pub(crate) fn mark_done(&mut self, done: Id<RuleList>) {
        self.tree[done].count += 1;
        let mut current = Some(done);

        while let Some(id) = current {
            let list = &mut self.tree[id];
            list.count -= 1;
            if list.count > 0 {
                break;
            }
            self.free.push(id);
            current = list.parent;
        }
    }
}
