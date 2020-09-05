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
 - start clauses: if no conjecture clauses are available, negative clauses are selected instead
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
<some thinking occurs...>
% SZS status Theorem for PUZ001+1
% SZS output begin CNFRefutation for PUZ001+1
cnf(c0, negated_conjecture,
	~killed(agatha,agatha)
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55)])).

cnf(c1, axiom,
	X0 = agatha | X0 = charles | X0 = butler | ~lives(X0)
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_3)])).
cnf(a0, assumption,
	agatha = agatha).
cnf(c2, plain,
	X0 = charles | X0 = butler | ~lives(X0),
	inference(strict_backward_paramodulation, [assumptions([a0]), status(thm)], [c1])).
cnf(c3, plain,
	X1 != X0 | ~killed(X1,agatha),
	inference(strict_backward_paramodulation, [assumptions([a0]), status(thm)], [c1])).

cnf(a1, assumption,
	X1 = X0).
cnf(c4, plain,
	~killed(X1,agatha),
	inference(reflexivity, [assumptions([a1]), status(thm)], [c3])).

cnf(c5, axiom,
	killed(sK0,agatha)
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_1)])).
cnf(a2, assumption,
	X1 = sK0).
cnf(a3, assumption,
	agatha = agatha).
cnf(c6, plain,
	$false,
	inference(strict_extension, [assumptions([a2, a3]), status(thm)], [c4, c5])).

cnf(c7, axiom,
	killed(sK0,agatha)
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_1)])).
cnf(a4, assumption,
	charles = X2).
cnf(a5, assumption,
	sK0 = X0).
cnf(c8, plain,
	X0 = butler | ~lives(X0),
	inference(strict_forward_paramodulation, [assumptions([a4, a5]), status(thm)], [c2, c7])).
cnf(c9, plain,
	killed(X2,agatha),
	inference(strict_forward_paramodulation, [assumptions([a4, a5]), status(thm)], [c2, c7])).

cnf(c10, axiom,
	~killed(X3,X4) | hates(X3,X4)
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_4)])).
cnf(a6, assumption,
	X2 = X3).
cnf(a7, assumption,
	agatha = X4).
cnf(c11, plain,
	hates(X3,X4),
	inference(strict_extension, [assumptions([a6, a7]), status(thm)], [c9, c10])).

cnf(c12, axiom,
	~hates(charles,X5) | ~hates(agatha,X5)
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_6)])).
cnf(a8, assumption,
	X3 = charles).
cnf(a9, assumption,
	X4 = X5).
cnf(c13, plain,
	~hates(agatha,X5),
	inference(strict_extension, [assumptions([a8, a9]), status(thm)], [c11, c12])).

cnf(c14, axiom,
	hates(agatha,X6) | X6 = butler
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_7)])).
cnf(a10, assumption,
	agatha = agatha).
cnf(a11, assumption,
	X5 = X6).
cnf(c15, plain,
	X6 = butler,
	inference(strict_extension, [assumptions([a10, a11]), status(thm)], [c13, c14])).

cnf(c16, axiom,
	agatha != butler
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_11)])).
cnf(a12, assumption,
	butler = X7).
cnf(a13, assumption,
	agatha = X6).
cnf(c17, plain,
	X7 != butler,
	inference(strict_forward_paramodulation, [assumptions([a12, a13]), status(thm)], [c15, c16])).

cnf(a14, assumption,
	X7 = butler).
cnf(c18, plain,
	$false,
	inference(reflexivity, [assumptions([a14]), status(thm)], [c17])).

cnf(c19, axiom,
	killed(sK0,agatha)
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_1)])).
cnf(a15, assumption,
	butler = X8).
cnf(a16, assumption,
	sK0 = X0).
cnf(c20, plain,
	~lives(X0),
	inference(strict_forward_paramodulation, [assumptions([a15, a16]), status(thm)], [c8, c19])).
cnf(c21, plain,
	killed(X8,agatha),
	inference(strict_forward_paramodulation, [assumptions([a15, a16]), status(thm)], [c8, c19])).

cnf(c22, axiom,
	~killed(X9,X10) | ~richer(X9,X10)
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_5)])).
cnf(a17, assumption,
	X8 = X9).
cnf(a18, assumption,
	agatha = X10).
cnf(c23, plain,
	~richer(X9,X10),
	inference(strict_extension, [assumptions([a17, a18]), status(thm)], [c21, c22])).

cnf(c24, axiom,
	richer(X11,agatha) | hates(butler,X11)
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_8)])).
cnf(a19, assumption,
	X9 = X11).
cnf(a20, assumption,
	X10 = agatha).
cnf(c25, plain,
	hates(butler,X11),
	inference(strict_extension, [assumptions([a19, a20]), status(thm)], [c23, c24])).

cnf(c26, axiom,
	~hates(X12,sK1(X12))
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_10)])).
cnf(a21, assumption,
	X13 = butler).
cnf(a22, assumption,
	X14 = X11).
cnf(c27, plain,
	X12 != X13 | sK1(X12) != X14,
	inference(lazy_extension, [assumptions([a21, a22]), status(thm)], [c25, c26])).

cnf(a23, assumption,
	X12 = X13).
cnf(c28, plain,
	sK1(X12) != X14,
	inference(reflexivity, [assumptions([a23]), status(thm)], [c27])).

cnf(c29, axiom,
	X15 = butler | hates(agatha,X15)
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_7)])).
cnf(a24, assumption,
	sK1(X12) = X15).
cnf(c30, plain,
	hates(agatha,X15),
	inference(variable_backward_paramodulation, [assumptions([a24]), status(thm)], [c28, c29])).
cnf(c31, plain,
	butler != X16 | X16 != X14,
	inference(variable_backward_paramodulation, [assumptions([a24]), status(thm)], [c28, c29])).

cnf(a25, assumption,
	butler = X16).
cnf(c32, plain,
	X16 != X14,
	inference(reflexivity, [assumptions([a25]), status(thm)], [c31])).

cnf(a26, assumption,
	X16 = X14).
cnf(c33, plain,
	$false,
	inference(reflexivity, [assumptions([a26]), status(thm)], [c32])).

cnf(c34, axiom,
	~hates(agatha,X17) | hates(butler,X17)
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_9)])).
cnf(a27, assumption,
	agatha = agatha).
cnf(a28, assumption,
	X15 = X17).
cnf(c35, plain,
	hates(butler,X17),
	inference(strict_extension, [assumptions([a27, a28]), status(thm)], [c30, c34])).

cnf(c36, lemma,
	~hates(X12,sK1(X12))).
cnf(a29, assumption,
	butler = X12).
cnf(a30, assumption,
	X17 = sK1(X12)).
cnf(c37, plain,
	$false,
	inference(reduction, [assumptions([a29, a30]), status(thm)], [c35, c36])).

cnf(c38, axiom,
	lives(sK0)
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_1)])).
cnf(a31, assumption,
	X0 = sK0).
cnf(c39, plain,
	$false,
	inference(strict_extension, [assumptions([a31]), status(thm)], [c20, c38])).

cnf(c40, plain,
	$false,
	inference(constraint_solving, [
		bind(X0, sK0),
		bind(X1, sK0),
		bind(X2, charles),
		bind(X3, charles),
		bind(X4, agatha),
		bind(X5, agatha),
		bind(X6, agatha),
		bind(X7, butler),
		bind(X8, butler),
		bind(X9, butler),
		bind(X10, agatha),
		bind(X11, butler),
		bind(X12, butler),
		bind(X13, butler),
		bind(X14, butler),
		bind(X15, sK1(X12)),
		bind(X16, butler),
		bind(X17, sK1(X12))
	],
	[a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31])).

% SZS output end CNFRefutation for PUZ001+1
```

## References
[1] Paskevich, Andrei. "Connection tableaux with lazy paramodulation." Journal of Automated Reasoning 40.2-3 (2008): 179-194.
[2] Otten, Jens. "Restricting backtracking in connection calculi." AI Communications 23.2-3 (2010): 159-182.
