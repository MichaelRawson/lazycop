use crate::prelude::*;
use std::collections::HashMap;
use std::mem;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Value {
    A,
    B,
    N,
    Top(Id<Symbol>),
}

fn fp4m(
    symbol_list: &SymbolList,
    term_list: &TermList,
    term: Id<Term>,
) -> [Value; 4] {
    let mut result = [Value::B, Value::B, Value::B, Value::B];
    match term_list.view(symbol_list, term) {
        TermView::Variable => {
            result[0] = Value::A;
        }
        TermView::Function(f, mut args) => {
            result[0] = Value::Top(f);
            if let Some(arg1) = args.next() {
                match term_list.view(symbol_list, arg1) {
                    TermView::Variable => {
                        result[1] = Value::A;
                    }
                    TermView::Function(f1, mut args1) => {
                        result[1] = Value::Top(f1);
                        if let Some(arg11) = args1.next() {
                            match term_list.view(symbol_list, arg11) {
                                TermView::Variable => {
                                    result[3] = Value::A;
                                }
                                TermView::Function(f11, _) => {
                                    result[3] = Value::Top(f11);
                                }
                            }
                        } else {
                            result[3] = Value::N;
                        }
                    }
                }
                if let Some(arg2) = args.next() {
                    match term_list.view(symbol_list, arg2) {
                        TermView::Variable => {
                            result[2] = Value::A;
                        }
                        TermView::Function(f2, _) => {
                            result[2] = Value::Top(f2);
                        }
                    }
                } else {
                    result[2] = Value::N;
                }
            } else {
                result[1] = Value::N;
                result[2] = Value::N;
                result[3] = Value::N;
            }
        }
    }
    result
}

struct Node;

pub struct Index<T> {
    branches: Vec<Vec<(Value, Id<Node>)>>,
    stored: HashMap<Id<Node>, T>,
}

impl<T> Default for Index<T> {
    fn default() -> Self {
        let branches = vec![vec![]];
        let stored = HashMap::new();
        Self { branches, stored }
    }
}

impl<T: Default> Index<T> {
    pub fn make_entry(
        &mut self,
        symbol_list: &SymbolList,
        term_list: &TermList,
        term: Id<Term>,
    ) -> &mut T {
        let fingerprint = fp4m(symbol_list, term_list, term);

        let mut current: Id<Node> = 0.into();
        for value in &fingerprint {
            let children = &mut self.branches[current.index()];
            current = children
                .binary_search_by_key(value, |(value, _)| *value)
                .map(|index| children[index].1)
                .unwrap_or_else(|index| {
                    let new_id = self.branches.len().into();
                    self.branches.push(vec![]);
                    let entry = (*value, new_id);
                    self.branches[current.index()].insert(index, entry);
                    new_id
                });
        }
        self.stored.entry(current).or_default()
    }
}

impl<T> Index<T> {
    pub fn query_unifiable(
        &self,
        symbol_list: &SymbolList,
        term_list: &TermList,
        term: Id<Term>,
    ) -> Vec<&T> {
        let fingerprint = fp4m(symbol_list, term_list, term);

        let mut current_nodes = vec![0.into()];
        let mut next_nodes = vec![];
        for value in &fingerprint {
            for node in &current_nodes {
                self.collect_unifiable_nodes(&mut next_nodes, *node, *value);
            }
            mem::swap(&mut current_nodes, &mut next_nodes);
            next_nodes.clear();
        }

        current_nodes
            .iter()
            .filter_map(|node| self.stored.get(node))
            .collect()
    }

    fn collect_unifiable_nodes(
        &self,
        results: &mut Vec<Id<Node>>,
        node: Id<Node>,
        value: Value,
    ) {
        let children = &self.branches[node.index()];
        let mut add_if_present = |value| {
            if let Ok(index) =
                children.binary_search_by_key(&value, |(value, _)| *value)
            {
                results.push(children[index].1);
            }
        };

        match value {
            Value::A => {
                results.extend(
                    children
                        .iter()
                        .filter(|(value, _)| *value != Value::N)
                        .map(|(_, next)| next),
                );
            }
            Value::B => {
                results.extend(children.iter().map(|(_, next)| next));
            }
            Value::N => {
                add_if_present(Value::B);
                add_if_present(Value::N);
            }
            Value::Top(_) => {
                add_if_present(value);
                add_if_present(Value::A);
                add_if_present(Value::B);
            }
        }
    }
}
