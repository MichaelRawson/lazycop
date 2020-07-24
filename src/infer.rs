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
    let clause = tableau.current_clause();
    let literal = &literals[clause.current_literal()];
    if literal.is_predicate() {
        predicate_rules(possible, tableau, problem, terms, literals, literal);
    } else if literal.is_equality() {
        equality_rules(possible, tableau, problem, terms, literals, literal);
    }
    equality_reduction_rules(
        possible,
        tableau,
        &problem.symbols,
        terms,
        literals,
        literal,
    );
    equality_extension_rules(possible, problem, terms, literal);
}

fn predicate_rules<E: Extend<Rule>>(
    possible: &mut E,
    tableau: &Tableau,
    problem: &Problem,
    terms: &Terms,
    literals: &Literals,
    literal: &Literal,
) {
    predicate_reduction_rules(possible, tableau, terms, literals, literal);
    predicate_extension_rules(possible, problem, terms, literal);
}

fn predicate_reduction_rules<E: Extend<Rule>>(
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
            .map(|literal| PredicateReduction { literal })
            .map(Rule::PredicateReduction),
    );
}

fn predicate_extension_rules<E: Extend<Rule>>(
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
        .map(|occurrence| PredicateExtension { occurrence });

    for extension in extensions {
        if problem.has_equality {
            possible.extend(once(Rule::LazyPredicateExtension(extension)));
        }
        possible.extend(once(Rule::StrictPredicateExtension(extension)));
    }
}

fn equality_reduction_rules<E: Extend<Rule>>(
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
                equality_reduction(terms, id, target, left)
                    .map(Rule::LREqualityReduction),
            );
            possible.extend(
                equality_reduction(terms, id, target, right)
                    .map(Rule::RLEqualityReduction),
            );
        });
    }
}

fn equality_extension_rules<E: Extend<Rule>>(
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
                .map(|occurrence| EqualityExtension { target, occurrence })
                .map(Rule::VariableExtension),
        );

        let symbol = terms.symbol(target);
        let function_extensions = problem
            .index
            .query_function_equalities(symbol)
            .map(|occurrence| EqualityExtension { target, occurrence });
        for extension in function_extensions {
            possible.extend(once(Rule::LazyFunctionExtension(extension)));
            possible.extend(once(Rule::StrictFunctionExtension(extension)));
        }
    });
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
        subterm_reductions(
            possible,
            tableau,
            &problem.symbols,
            terms,
            literals,
            literal,
        );
        subterm_extensions(possible, problem, terms, literal);
    }
}

fn subterm_reductions<E: Extend<Rule>>(
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
                equality_reduction(terms, id, target, left)
                    .map(Rule::LRSubtermReduction),
            );
            possible.extend(
                equality_reduction(terms, id, target, right)
                    .map(Rule::RLSubtermReduction),
            );
        });
    }
}

fn subterm_extensions<E: Extend<Rule>>(
    possible: &mut E,
    problem: &Problem,
    terms: &Terms,
    literal: &Literal,
) {
    let (left, right) = literal.get_equality();
    subterm_extensions_one_sided(possible, problem, terms, left, true);
    subterm_extensions_one_sided(possible, problem, terms, right, false);
}

fn subterm_extensions_one_sided<E: Extend<Rule>>(
    possible: &mut E,
    problem: &Problem,
    terms: &Terms,
    from: Id<Term>,
    lr: bool,
) {
    if terms.is_variable(from) {
        for occurrence in problem.index.query_all_subterms() {
            subterm_extensions_single(possible, occurrence, lr);
        }
    } else {
        for occurrence in problem.index.query_subterms(terms.symbol(from)) {
            subterm_extensions_single(possible, occurrence, lr);
        }
    }
}

fn subterm_extensions_single<E: Extend<Rule>>(
    possible: &mut E,
    occurrence: Id<SubtermOccurrence>,
    lr: bool,
) {
    let extension = SubtermExtension { occurrence };
    if lr {
        possible.extend(once(Rule::LRStrictSubtermExtension(extension)));
        possible.extend(once(Rule::LRLazySubtermExtension(extension)));
    } else {
        possible.extend(once(Rule::RLStrictSubtermExtension(extension)));
        possible.extend(once(Rule::RLLazySubtermExtension(extension)));
    }
}

fn equality_reduction(
    terms: &Terms,
    literal: Id<Literal>,
    target: Id<Term>,
    from: Id<Term>,
) -> Option<EqualityReduction> {
    if terms.is_variable(from) || terms.symbol(from) == terms.symbol(target) {
        Some(EqualityReduction { literal, target })
    } else {
        None
    }
}
