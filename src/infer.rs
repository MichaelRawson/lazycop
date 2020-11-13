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
    bindings: &Bindings,
) {
    if tableau.is_empty() {
        start_rules(possible, problem);
        return;
    }

    let clause = tableau.current_clause();
    let literal = &literals[clause.current_literal()];
    if literal.is_predicate() {
        predicate_rules(
            possible, tableau, problem, terms, literals, bindings, literal,
        );
    } else if literal.is_equality() {
        equality_rules(
            possible, tableau, problem, terms, literals, bindings, literal,
        );
    }
    forward_demodulation_rules(
        possible,
        tableau,
        &problem.symbols,
        terms,
        literals,
        bindings,
        literal,
    );
    backward_paramodulation_rules(possible, problem, terms, bindings, literal);
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
    bindings: &Bindings,
    literal: &Literal,
) {
    for id in tableau.reduction_literals() {
        let reduction = &literals[id];
        if !reduction.polarity || !reduction.is_equality() {
            continue;
        }
        let (left, right) = reduction.get_equality();
        let left = bindings.resolve(terms, left);
        let right = bindings.resolve(terms, right);
        literal.subterms(symbols, terms, &mut |target| {
            possible.extend(
                demodulation(symbols, terms, bindings, id, target, left)
                    .map(Rule::LRForwardDemodulation),
            );
            possible.extend(
                demodulation(symbols, terms, bindings, id, target, right)
                    .map(Rule::RLForwardDemodulation),
            );
        });
    }
}

fn backward_paramodulation_rules<E: Extend<Rule>>(
    possible: &mut E,
    problem: &Problem,
    terms: &Terms,
    bindings: &Bindings,
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
        let backward_paramodulations =
            problem.index.query_function_equalities(symbol);
        for occurrence in backward_paramodulations {
            let extension = BackwardParamodulation { target, occurrence };
            possible.extend(once(Rule::LazyBackwardParamodulation(extension)));

            let occurrence = &problem.index.equality_occurrences[occurrence];
            let clause = &problem.clauses[occurrence.clause];
            let (left, right) =
                clause.literals[occurrence.literal].get_equality();
            let from = if occurrence.l2r { left } else { right };
            if external_match(
                &problem.symbols,
                terms,
                bindings,
                &clause.terms,
                target,
                from,
            ) {
                possible.extend(once(Rule::StrictBackwardParamodulation(
                    extension,
                )));
            }
        }
    });
}

fn predicate_rules<E: Extend<Rule>>(
    possible: &mut E,
    tableau: &Tableau,
    problem: &Problem,
    terms: &Terms,
    literals: &Literals,
    bindings: &Bindings,
    literal: &Literal,
) {
    reduction_rules(
        possible,
        tableau,
        &problem.symbols,
        terms,
        literals,
        bindings,
        literal,
    );
    extension_rules(possible, problem, terms, bindings, literal);
}

fn reduction_rules<E: Extend<Rule>>(
    possible: &mut E,
    tableau: &Tableau,
    symbols: &Symbols,
    terms: &Terms,
    literals: &Literals,
    bindings: &Bindings,
    literal: &Literal,
) {
    let polarity = literal.polarity;
    let predicate = literal.get_predicate();
    possible.extend(
        tableau
            .reduction_literals()
            .filter(|id| {
                let reduction = &literals[*id];
                reduction.polarity != polarity
                    && reduction.is_predicate()
                    && internal_match(
                        symbols,
                        terms,
                        bindings,
                        predicate,
                        reduction.get_predicate(),
                    )
            })
            .map(|literal| Reduction { literal })
            .map(Rule::Reduction),
    );
}

fn extension_rules<E: Extend<Rule>>(
    possible: &mut E,
    problem: &Problem,
    terms: &Terms,
    bindings: &Bindings,
    literal: &Literal,
) {
    let polarity = !literal.polarity;
    let symbol = literal.get_predicate_symbol(terms);
    let occurrences = problem.index.query_predicates(polarity, symbol);

    for occurrence in occurrences {
        let extension = Extension { occurrence };
        if problem.has_equality {
            possible.extend(once(Rule::LazyExtension(extension)));
        }

        let occurrence = &problem.index.predicate_occurrences[occurrence];
        let extension_clause = &problem.clauses[occurrence.clause];
        if external_match(
            &problem.symbols,
            terms,
            bindings,
            &extension_clause.terms,
            literal.get_predicate(),
            extension_clause.literals[occurrence.literal].get_predicate(),
        ) {
            possible.extend(once(Rule::StrictExtension(extension)));
        }
    }
}

fn equality_rules<E: Extend<Rule>>(
    possible: &mut E,
    tableau: &Tableau,
    problem: &Problem,
    terms: &Terms,
    literals: &Literals,
    bindings: &Bindings,
    literal: &Literal,
) {
    let (left, right) = literal.get_equality();
    let left = bindings.resolve(terms, left);
    let right = bindings.resolve(terms, right);
    if literal.polarity {
        if !terms.is_variable(left) && !terms.is_variable(right) {
            let f = terms.symbol(left);
            let g = terms.symbol(right);
            if f != g
                && problem.symbols[f].is_distinct_object()
                && problem.symbols[g].is_distinct_object()
            {
                possible.extend(once(Rule::DistinctObjects));
            }
        }
        backward_demodulation_rules(
            possible,
            tableau,
            &problem.symbols,
            terms,
            literals,
            bindings,
            left,
            right,
        );
        forward_paramodulation_rules(
            possible, problem, terms, bindings, left, right,
        );
    } else if internal_match(&problem.symbols, terms, bindings, left, right) {
        possible.extend(once(Rule::Reflexivity));
    }
}

fn backward_demodulation_rules<E: Extend<Rule>>(
    possible: &mut E,
    tableau: &Tableau,
    symbols: &Symbols,
    terms: &Terms,
    literals: &Literals,
    bindings: &Bindings,
    left: Id<Term>,
    right: Id<Term>,
) {
    for id in tableau.reduction_literals() {
        let reduction = &literals[id];
        reduction.subterms(symbols, terms, &mut |target| {
            possible.extend(
                demodulation(symbols, terms, bindings, id, target, left)
                    .map(Rule::LRBackwardDemodulation),
            );
            possible.extend(
                demodulation(symbols, terms, bindings, id, target, right)
                    .map(Rule::RLBackwardDemodulation),
            );
        });
    }
}

fn forward_paramodulation_rules<E: Extend<Rule>>(
    possible: &mut E,
    problem: &Problem,
    terms: &Terms,
    bindings: &Bindings,
    left: Id<Term>,
    right: Id<Term>,
) {
    forward_paramodulation_rules_one_sided(
        possible, problem, terms, bindings, left, true,
    );
    forward_paramodulation_rules_one_sided(
        possible, problem, terms, bindings, right, false,
    );
}

fn forward_paramodulation_rules_one_sided<E: Extend<Rule>>(
    possible: &mut E,
    problem: &Problem,
    terms: &Terms,
    bindings: &Bindings,
    from: Id<Term>,
    lr: bool,
) {
    if terms.is_variable(from) {
        for occurrence in problem.index.query_all_subterms() {
            let paramodulation = ForwardParamodulation { occurrence };
            if lr {
                possible.extend(once(Rule::LRStrictForwardParamodulation(
                    paramodulation,
                )));
                possible.extend(once(Rule::LRLazyForwardParamodulation(
                    paramodulation,
                )));
            } else {
                possible.extend(once(Rule::RLStrictForwardParamodulation(
                    paramodulation,
                )));
                possible.extend(once(Rule::RLLazyForwardParamodulation(
                    paramodulation,
                )));
            }
        }
    } else {
        for occurrence in problem.index.query_subterms(terms.symbol(from)) {
            let paramodulation = ForwardParamodulation { occurrence };
            if lr {
                possible.extend(once(Rule::LRLazyForwardParamodulation(
                    paramodulation,
                )));
            } else {
                possible.extend(once(Rule::RLLazyForwardParamodulation(
                    paramodulation,
                )));
            }

            let occurrence = &problem.index.subterm_occurrences[occurrence];
            let clause = &problem.clauses[occurrence.clause];
            if external_match(
                &problem.symbols,
                terms,
                bindings,
                &clause.terms,
                from,
                occurrence.subterm,
            ) {
                if lr {
                    possible.extend(once(
                        Rule::LRStrictForwardParamodulation(paramodulation),
                    ));
                } else {
                    possible.extend(once(
                        Rule::RLStrictForwardParamodulation(paramodulation),
                    ));
                }
            }
        }
    }
}

fn demodulation(
    symbols: &Symbols,
    terms: &Terms,
    bindings: &Bindings,
    literal: Id<Literal>,
    target: Id<Term>,
    from: Id<Term>,
) -> Option<Demodulation> {
    if internal_match(symbols, terms, bindings, target, from) {
        Some(Demodulation { literal, target })
    } else {
        None
    }
}

fn internal_match(
    symbols: &Symbols,
    terms: &Terms,
    bindings: &Bindings,
    left: Id<Term>,
    right: Id<Term>,
) -> bool {
    match (terms.view(symbols, left), terms.view(symbols, right)) {
        (TermView::Variable(_), _) | (_, TermView::Variable(_)) => true,
        (TermView::Function(f, ss), TermView::Function(g, ts)) if f == g => {
            ss.into_iter().zip(ts.into_iter()).all(|(s, t)| {
                internal_match(
                    symbols,
                    terms,
                    bindings,
                    bindings.resolve(terms, terms.resolve(s)),
                    bindings.resolve(terms, terms.resolve(t)),
                )
            })
        }
        (TermView::Function(_, _), TermView::Function(_, _)) => false,
    }
}

fn external_match(
    symbols: &Symbols,
    terms: &Terms,
    bindings: &Bindings,
    external_terms: &Terms,
    left: Id<Term>,
    right: Id<Term>,
) -> bool {
    match (
        terms.view(symbols, left),
        external_terms.view(symbols, right),
    ) {
        (TermView::Variable(_), _) | (_, TermView::Variable(_)) => true,
        (TermView::Function(f, ss), TermView::Function(g, ts)) if f == g => {
            ss.into_iter().zip(ts.into_iter()).all(|(s, t)| {
                external_match(
                    symbols,
                    terms,
                    bindings,
                    external_terms,
                    terms.resolve(s),
                    external_terms.resolve(t),
                )
            })
        }
        (TermView::Function(_, _), TermView::Function(_, _)) => false,
    }
}
