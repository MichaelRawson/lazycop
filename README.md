<p align="center">
 <a href="https://github.com/MichaelRawson/lazycop">
  <img src="https://github.com/MichaelRawson/lazycop/blob/master/etc/logo.svg">
 </a>
</p>

# lazyCoP
lazyCoP is an automatic theorem prover for first-order logic with equality.
It implements the lazy paramodulation calculus [1], an extension of the predicate connection tableau calculus to handle equality reasoning with ordered paramodulation.
lazyCoP also implements the following well-known refinements of the predicate calculus:
 - definitional clause normal form translation, as in [2]
 - relevancy information: lazyCoP will start with clauses related to the conjecture if present
 - tautology elimination: no clause added to the tableau may contain some kinds of tautology
 - path regularity: no literal can occur twice in any path
 - enforced _hochklappen_ ("folding up"): refutations of unit literal "lemmas" are available for re-use where possible
 - strong regularity: a branch cannot be extended if the leaf predicate could be closed by reduction with a path literal or lemma

See the _Handbook of Automated Reasoning (Vol. II)_, "Model Elimination and Connection Tableau Procedures" for more information and an introduction to the predicate calculus.

As well as these predicate refinements, lazyCoP implements some obvious modifications for equality literals:
 - reflexivity elimination: no clause may contain `s = t` if `s`, `t` are identical
 - reflexive regularity: if `s != t` is a literal in the tableau and `s`, `t` are identical, it must be closed immediately
 - symmetric path regularity: neither `s = t` nor `t = s` may appear in a path if `s = t` does

One practical issue with the lazy paramodulation calculus is that proofs may be significantly longer, particularly if "lazy" steps could in fact be "strict".
lazyCoP avoids this by implementing both lazy and strict versions of all lazy inferences.
The resulting duplication is eliminated by refinement: lazy inferences are not permitted to simulate strict rule application.

The term ordering employed is the lexicographic path ordering.
For symbol precedence, `f > g` iff either `f` has a larger arity than `g`, or the two have equal arity but `f` appears later in the problem than `g`.

## Completeness
Paskevich claims [1] that the pure lazy paramodulation calculus is complete, but leaves demonstrating the completeness of refinements such as path regularity as future work.
lazyCoP implements a number of refinements of the calculus, all of which improve prover performance and appear to preserve completeness, at least empirically.
However, we are suspicious of the refinements' effect on completeness.

We are very interested in any true statements for which lazyCoP either terminates, or fails to find an "easy" proof.

## License
lazyCoP is MIT licensed. At the time of writing, we believe this to be compatible with all direct dependencies.

## Building
lazyCoP is written entirely in [Rust](https://rust-lang.org).
Therefore,
```
$ cargo build --release
```
will work as expected.
In order to get a [StarExec](https://starexec.org)-compatible, statically-linked solver, `etc/starexec.sh` will attempt such a build via linking to [MUSL](https://musl.libc.org/).
You _will_ need to edit this script a little (be careful!), but it should give you an idea of what's involved.
Cross-compiling to many different platforms is in principle possible.
Please let me know if you succeed in doing this!

## Usage
lazyCoP understands the [TPTP](http://tptp.org) FOF and CNF dialects and writes proofs in the corresponding [TSTP](http://www.tptp.org/TSTP/) output format.
`lazycop --help` will print a help message.
After a problem has been loaded successfully and preprocessed, lazyCoP will attempt to prove it.

One of three things will then happen:
 1. lazyCoP finds a proof, which it will print to standard output. Hooray!
 2. lazyCoP exhausts its search space without finding a proof. This means that the conjecture does not follow from the axioms, or that the set of clauses is satisfiable.
 3. lazyCoP runs until you get bored. If a proof exists, it will be found eventually, but perhaps not in reasonable time.

A typical run on [PUZ001+1](http://tptp.org/cgi-bin/SeeTPTP?Category=Problems&Domain=PUZ&File=PUZ001+1.p):
```
$ lazycop Problems/PUZ/PUZ001+1.p
% SZS status Theorem for PUZ001+1
% SZS output begin CNFRefutation for PUZ001+1
cnf(c0, negated_conjecture,
	~killed(agatha,agatha)
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55)])).

cnf(c1, axiom,
	X0 = agatha | X0 = charles | X0 = butler | ~lives(X0)
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_3)])).
cnf(c2, plain,
	X0 = charles | X0 = butler | ~lives(X0),
	inference(strict_backward_paramodulation, [status(thm)], [c0, c1])).
cnf(c3, plain,
	X1 != X0 | ~killed(agatha,X1),
	inference(strict_backward_paramodulation, [status(thm)], [c0, c1])).

cnf(c4, plain,
	~killed(agatha,X1),
	inference(reflexivity, [status(thm), bind(X1,X0)], [c3])).

cnf(c5, lemma,
	X0 = agatha).
cnf(c6, plain,
	~killed(X2,X1),
	inference(forward_demodulation, [status(thm), bind(X2,X0)], [c4, c5])).

cnf(c7, axiom,
	killed(sK0,agatha)
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_1)])).
cnf(c8, plain,
	X3 != sK0 | X4 != agatha,
	inference(lazy_extension, [status(thm), bind(X3,X0), bind(X4,X0)], [c6, c7])).

cnf(c9, plain,
	X4 != agatha,
	inference(reflexivity, [status(thm), bind(X0,sK0), bind(X1,sK0), bind(X2,sK0), bind(X3,sK0), bind(X4,sK0)], [c8])).

cnf(c10, lemma,
	X0 = agatha).
cnf(c11, plain,
	X4 != X5,
	inference(forward_demodulation, [status(thm), bind(X5,sK0)], [c9, c10])).

cnf(c12, plain,
	$false,
	inference(reflexivity, [status(thm)], [c11])).

cnf(c13, axiom,
	killed(sK0,agatha)
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_1)])).
cnf(c14, plain,
	X0 = butler | ~lives(X0),
	inference(strict_forward_paramodulation, [status(thm), bind(X6,charles)], [c2, c13])).
cnf(c15, plain,
	killed(X6,agatha),
	inference(strict_forward_paramodulation, [status(thm), bind(X6,charles)], [c2, c13])).

cnf(c16, axiom,
	~killed(X7,X8) | hates(X7,X8)
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_4)])).
cnf(c17, plain,
	hates(X7,X8),
	inference(strict_extension, [status(thm), bind(X7,charles), bind(X8,agatha)], [c15, c16])).

cnf(c18, axiom,
	~hates(charles,X9) | ~hates(agatha,X9)
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_6)])).
cnf(c19, plain,
	~hates(agatha,X9),
	inference(strict_extension, [status(thm), bind(X9,agatha)], [c17, c18])).

cnf(c20, axiom,
	hates(agatha,X10) | X10 = butler
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_7)])).
cnf(c21, plain,
	X10 = butler,
	inference(strict_extension, [status(thm), bind(X10,agatha)], [c19, c20])).

cnf(c22, axiom,
	agatha != butler
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_11)])).
cnf(c23, plain,
	X11 != butler,
	inference(strict_forward_paramodulation, [status(thm), bind(X11,butler)], [c21, c22])).

cnf(c24, plain,
	$false,
	inference(reflexivity, [status(thm)], [c23])).

cnf(c25, axiom,
	killed(sK0,agatha)
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_1)])).
cnf(c26, plain,
	~lives(X0),
	inference(strict_forward_paramodulation, [status(thm), bind(X12,butler)], [c14, c25])).
cnf(c27, plain,
	killed(X12,agatha),
	inference(strict_forward_paramodulation, [status(thm), bind(X12,butler)], [c14, c25])).

cnf(c28, axiom,
	~killed(X13,X14) | ~richer(X13,X14)
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_5)])).
cnf(c29, plain,
	~richer(X13,X14),
	inference(strict_extension, [status(thm), bind(X13,butler), bind(X14,agatha)], [c27, c28])).

cnf(c30, axiom,
	richer(X15,agatha) | hates(butler,X15)
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_8)])).
cnf(c31, plain,
	hates(butler,X15),
	inference(strict_extension, [status(thm), bind(X15,butler)], [c29, c30])).

cnf(c32, axiom,
	~hates(X16,sK1(X16))
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_10)])).
cnf(c33, plain,
	X17 != X16 | X18 != sK1(X16),
	inference(lazy_extension, [status(thm), bind(X17,butler), bind(X18,butler)], [c31, c32])).

cnf(c34, plain,
	X18 != sK1(X16),
	inference(reflexivity, [status(thm), bind(X16,butler)], [c33])).

cnf(c35, axiom,
	X19 = butler | hates(agatha,X19)
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_7)])).
cnf(c36, plain,
	hates(agatha,X19),
	inference(variable_backward_paramodulation, [status(thm), bind(X19,sK1(X16))], [c34, c35])).
cnf(c37, plain,
	X20 != butler | X18 != X20,
	inference(variable_backward_paramodulation, [status(thm), bind(X19,sK1(X16))], [c34, c35])).

cnf(c38, plain,
	X18 != X20,
	inference(reflexivity, [status(thm), bind(X20,butler)], [c37])).

cnf(c39, plain,
	$false,
	inference(reflexivity, [status(thm)], [c38])).

cnf(c40, axiom,
	~hates(agatha,X21) | hates(butler,X21)
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_9)])).
cnf(c41, plain,
	hates(butler,X21),
	inference(strict_extension, [status(thm), bind(X21,sK1(X16))], [c36, c40])).

cnf(c42, lemma,
	~hates(X16,sK1(X16))).
cnf(c43, plain,
	$false,
	inference(reduction, [status(thm)], [c41, c42])).

cnf(c44, axiom,
	lives(sK0)
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_1)])).
cnf(c45, plain,
	$false,
	inference(strict_extension, [status(thm)], [c26, c44])).

% SZS output end CNFRefutation for PUZ001+1
$
```

## References
1. Paskevich, Andrei. "Connection tableaux with lazy paramodulation." Journal of Automated Reasoning 40.2-3 (2008): 179-194.
2. Otten, Jens. "Restricting backtracking in connection calculi." AI Communications 23.2-3 (2010): 159-182.
