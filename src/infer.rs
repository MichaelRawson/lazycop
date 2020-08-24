use crate::index::*;
use crate::prelude::*;
use crate::rule::*;
use crate::tableau::Tableau;
use std::iter::once;

pub(crate) fn rules<E: Extend<Rule>>(
    possible: &mut E,
    tableau: &Tableau,
    problem: &Problem,
    terms: &Terms,
    literals: &Literals,
) {
    if tableau.is_empty() {
        start_rules(possible, problem);
        return;
    }

    let clause = tableau.current_clause();
    let literal = &literals[clause.current_literal()];
    if literal.is_predicate() {
        predicate_rules(possible, tableau, problem, terms, literals, literal);
    } else if literal.is_equality() {
        equality_rules(possible, tableau, problem, terms, literals, literal);
    }
    forward_demodulation_rules(
        possible,
        tableau,
        &problem.symbols,
        terms,
        literals,
        literal,
    );
    backward_paramodulation_rules(possible, problem, terms, literal);
}

fn start_rules<E: Extend<Rule>>(possible: &mut E, problem: &Problem) {
    possible.extend(
        problem
            .start_clauses
            .iter()
            .copied()
            .map(|clause| Start { clause })
            .map(Rule::Start),
    );
}

fn forward_demodulation_rules<E: Extend<Rule>>(
    possible: &mut E,
    tableau: &Tableau,
    symbols: &Symbols,
    terms: &Terms,
    literals: &Literals,
    literal: &Literal,
) {
    for id in tableau.reduction_literals() {
        let reduction = &literals[id];
        if !reduction.polarity || !reduction.is_equality() {
            continue;
        }
        let (left, right) = reduction.get_equality();
        literal.subterms(symbols, terms, &mut |target| {
            possible.extend(
                demodulation(terms, id, target, left)
                    .map(Rule::LRForwardDemodulation),
            );
            possible.extend(
                demodulation(terms, id, target, right)
                    .map(Rule::RLForwardDemodulation),
            );
        });
    }
}

fn backward_paramodulation_rules<E: Extend<Rule>>(
    possible: &mut E,
    problem: &Problem,
    terms: &Terms,
    literal: &Literal,
) {
    literal.subterms(&problem.symbols, terms, &mut |target| {
        possible.extend(
            problem
                .index
                .query_variable_equalities()
                .map(|occurrence| BackwardParamodulation {
                    target,
                    occurrence,
                })
                .map(Rule::VariableBackwardParamodulation),
        );

        let symbol = terms.symbol(target);
        let backward_paramodulations = problem
            .index
            .query_function_equalities(symbol)
            .map(|occurrence| BackwardParamodulation { target, occurrence });
        for extension in backward_paramodulations {
            possible.extend(once(Rule::LazyBackwardParamodulation(extension)));
            possible
                .extend(once(Rule::StrictBackwardParamodulation(extension)));
        }
    });
}

fn predicate_rules<E: Extend<Rule>>(
    possible: &mut E,
    tableau: &Tableau,
    problem: &Problem,
    terms: &Terms,
    literals: &Literals,
    literal: &Literal,
) {
    reduction_rules(possible, tableau, terms, literals, literal);
    extension_rules(possible, problem, terms, literal);
}

fn reduction_rules<E: Extend<Rule>>(
    possible: &mut E,
    tableau: &Tableau,
    terms: &Terms,
    literals: &Literals,
    literal: &Literal,
) {
    let polarity = literal.polarity;
    let symbol = literal.get_predicate_symbol(terms);
    possible.extend(
        tableau
            .reduction_literals()
            .filter(|id| {
                let reduction = &literals[*id];
                reduction.polarity != polarity
                    && reduction.is_predicate()
                    && reduction.get_predicate_symbol(terms) == symbol
            })
            .map(|literal| Reduction { literal })
            .map(Rule::Reduction),
    );
}

fn extension_rules<E: Extend<Rule>>(
    possible: &mut E,
    problem: &Problem,
    terms: &Terms,
    literal: &Literal,
) {
    let polarity = !literal.polarity;
    let symbol = literal.get_predicate_symbol(terms);
    let extensions = problem
        .index
        .query_predicates(polarity, symbol)
        .map(|occurrence| Extension { occurrence });

    for extension in extensions {
        if problem.has_equality {
            possible.extend(once(Rule::LazyExtension(extension)));
        }
        possible.extend(once(Rule::StrictExtension(extension)));
    }
}

fn equality_rules<E: Extend<Rule>>(
    possible: &mut E,
    tableau: &Tableau,
    problem: &Problem,
    terms: &Terms,
    literals: &Literals,
    literal: &Literal,
) {
    if !literal.polarity {
        possible.extend(once(Rule::Reflexivity));
    } else {
        backward_demodulations(
            possible,
            tableau,
            &problem.symbols,
            terms,
            literals,
            literal,
        );
        forward_paramodulations(possible, problem, terms, literal);
    }
}

fn backward_demodulations<E: Extend<Rule>>(
    possible: &mut E,
    tableau: &Tableau,
    symbols: &Symbols,
    terms: &Terms,
    literals: &Literals,
    literal: &Literal,
) {
    let (left, right) = literal.get_equality();
    for id in tableau.reduction_literals() {
        let reduction = &literals[id];
        reduction.subterms(symbols, terms, &mut |target| {
            possible.extend(
                demodulation(terms, id, target, left)
                    .map(Rule::LRBackwardDemodulation),
            );
            possible.extend(
                demodulation(terms, id, target, right)
                    .map(Rule::RLBackwardDemodulation),
            );
        });
    }
}

fn forward_paramodulations<E: Extend<Rule>>(
    possible: &mut E,
    problem: &Problem,
    terms: &Terms,
    literal: &Literal,
) {
    let (left, right) = literal.get_equality();
    forward_paramodulations_one_sided(possible, problem, terms, left, true);
    forward_paramodulations_one_sided(possible, problem, terms, right, false);
}

fn forward_paramodulations_one_sided<E: Extend<Rule>>(
    possible: &mut E,
    problem: &Problem,
    terms: &Terms,
    from: Id<Term>,
    lr: bool,
) {
    if terms.is_variable(from) {
        for occurrence in problem.index.query_all_subterms() {
            forward_paramodulations_single(possible, occurrence, lr);
        }
    } else {
        for occurrence in problem.index.query_subterms(terms.symbol(from)) {
            forward_paramodulations_single(possible, occurrence, lr);
        }
    }
}

fn forward_paramodulations_single<E: Extend<Rule>>(
    possible: &mut E,
    occurrence: Id<SubtermOccurrence>,
    lr: bool,
) {
    let paramodulation = ForwardParamodulation { occurrence };
    if lr {
        possible
            .extend(once(Rule::LRStrictForwardParamodulation(paramodulation)));
        possible
            .extend(once(Rule::LRLazyForwardParamodulation(paramodulation)));
    } else {
        possible
            .extend(once(Rule::RLStrictForwardParamodulation(paramodulation)));
        possible
            .extend(once(Rule::RLLazyForwardParamodulation(paramodulation)));
    }
}

fn demodulation(
    terms: &Terms,
    literal: Id<Literal>,
    target: Id<Term>,
    from: Id<Term>,
) -> Option<Demodulation> {
    if terms.is_variable(from) || terms.symbol(from) == terms.symbol(target) {
        Some(Demodulation { literal, target })
    } else {
        None
    }
}
