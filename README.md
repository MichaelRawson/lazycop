<p align="center">
 <a href="https://github.com/MichaelRawson/lazycop">
  <img src="https://github.com/MichaelRawson/lazycop/blob/master/etc/logo.svg">
 </a>
</p>

# lazyCoP
lazyCoP is an automatic theorem prover for first-order logic with equality.
It implements the lazy paramodulation calculus [1], an extension of the predicate connection tableau calculus to handle equality reasoning with ordered paramodulation.
lazyCoP also implements the following well-known refinements of the predicate calculus:
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
However, we are suspicious of the refinements' effect on completeness, particularly of superposition-style ordering constraints.

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
In order to get a [StarExec](https://starexec.org)-compatible solver, `etc/starexec.sh` will attempt to build one via cross-compiling to MUSL.
You _will_ need to edit this script a little (be careful!), but it should give you an idea of what's involved.
Cross-compiling to many different platforms is in principle possible.
Please let me know if you succeed in doing this!

## Usage

lazyCoP understands the [TPTP](http://tptp.org) FOF and CNF dialects and writes proofs in the corresponding [TSTP](http://www.tptp.org/TSTP/) output format.
`lazycop --help` will print a help message.
After a problem has been loaded successfully and preprocessed, lazyCoP will attempt to prove it.

One of three things will then happen:
 1. lazyCoP finds a proof, which it will print to standard output. Hooray!
 2. lazyCoP exhausts its search space without finding a proof. This means that the conjecture does not follow from the axioms, or that the set of clauses is satisfiable. This is reported `SZS status GaveUp`.
 3. lazyCoP runs until you get bored. If a proof exists, it will be found eventually, but perhaps not in reasonable time.
 
A typical run on [PUZ001+1](http://tptp.org/cgi-bin/SeeTPTP?Category=Problems&Domain=PUZ&File=PUZ001+1.p):
```
$ lazycop Problems/PUZ/PUZ001+1.p
<some thinking occurs...>
% SZS status Unsatisfiable for PUZ001+1
% SZS output begin CNFRefutation for PUZ001+1
cnf(c0, negated_conjecture,
	~killed(agatha,agatha)
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55)])).
cnf(c1, plain,
	~killed(agatha,agatha),
	inference(start, [], [c0])).

cnf(c2, axiom,
	X0 = charles | X0 = butler | X0 = agatha | ~lives(X0)
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_3)])).
cnf(a0, assumption,
	agatha = agatha).
cnf(c3, plain,
	X1 = charles | X1 = butler | ~lives(X1),
	inference(strict_backward_paramodulation, [assumptions([a0])], [c1, c2])).
cnf(c4, plain,
	X2 != X1 | ~killed(X2,agatha),
	inference(strict_backward_paramodulation, [assumptions([a0])], [c1, c2])).

cnf(a1, assumption,
	X2 = X1).
cnf(c5, plain,
	~killed(X2,agatha),
	inference(reflexivity, [assumptions([a1])], [c4])).

cnf(c6, axiom,
	killed(sK0,agatha)
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_1)])).
cnf(a2, assumption,
	X2 = sK0).
cnf(a3, assumption,
	agatha = agatha).
cnf(c7, plain,
	$false,
	inference(strict_extension, [assumptions([a2, a3])], [c5, c6])).

cnf(c8, axiom,
	killed(sK0,agatha)
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_1)])).
cnf(a4, assumption,
	charles = X3).
cnf(a5, assumption,
	sK0 = X1).
cnf(c9, plain,
	X1 = butler | ~lives(X1),
	inference(strict_forward_paramodulation, [assumptions([a4, a5])], [c3, c8])).
cnf(c10, plain,
	killed(X3,agatha),
	inference(strict_forward_paramodulation, [assumptions([a4, a5])], [c3, c8])).

cnf(c11, axiom,
	hates(X0,X4) | ~killed(X0,X4)
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_4)])).
cnf(a6, assumption,
	X3 = X5).
cnf(a7, assumption,
	agatha = X6).
cnf(c12, plain,
	hates(X5,X6),
	inference(strict_extension, [assumptions([a6, a7])], [c10, c11])).

cnf(c13, axiom,
	~hates(charles,X4) | ~hates(agatha,X4)
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_6)])).
cnf(a8, assumption,
	X5 = charles).
cnf(a9, assumption,
	X6 = X7).
cnf(c14, plain,
	~hates(agatha,X7),
	inference(strict_extension, [assumptions([a8, a9])], [c12, c13])).

cnf(c15, axiom,
	hates(agatha,X4) | X4 = butler
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_7)])).
cnf(a10, assumption,
	agatha = agatha).
cnf(a11, assumption,
	X7 = X8).
cnf(c16, plain,
	X8 = butler,
	inference(strict_extension, [assumptions([a10, a11])], [c14, c15])).

cnf(c17, axiom,
	agatha != butler
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_11)])).
cnf(a12, assumption,
	butler = X9).
cnf(a13, assumption,
	agatha = X8).
cnf(c18, plain,
	X9 != butler,
	inference(strict_forward_paramodulation, [assumptions([a12, a13])], [c16, c17])).

cnf(a14, assumption,
	X9 = butler).
cnf(c19, plain,
	$false,
	inference(reflexivity, [assumptions([a14])], [c18])).

cnf(c20, axiom,
	killed(sK0,agatha)
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_1)])).
cnf(a15, assumption,
	butler = X10).
cnf(a16, assumption,
	sK0 = X1).
cnf(c21, plain,
	~lives(X1),
	inference(strict_forward_paramodulation, [assumptions([a15, a16])], [c9, c20])).
cnf(c22, plain,
	killed(X10,agatha),
	inference(strict_forward_paramodulation, [assumptions([a15, a16])], [c9, c20])).

cnf(c23, axiom,
	~richer(X0,X4) | ~killed(X0,X4)
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_5)])).
cnf(a17, assumption,
	X10 = X11).
cnf(a18, assumption,
	agatha = X12).
cnf(c24, plain,
	~richer(X11,X12),
	inference(strict_extension, [assumptions([a17, a18])], [c22, c23])).

cnf(c25, axiom,
	hates(butler,X4) | richer(X4,agatha)
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_8)])).
cnf(a19, assumption,
	X11 = X13).
cnf(a20, assumption,
	X12 = agatha).
cnf(c26, plain,
	hates(butler,X13),
	inference(strict_extension, [assumptions([a19, a20])], [c24, c25])).

cnf(c27, axiom,
	~hates(X0,sK1(X0))
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_10)])).
cnf(a21, assumption,
	X14 = butler).
cnf(a22, assumption,
	X15 = X13).
cnf(c28, plain,
	X16 != X14 | sK1(X16) != X15,
	inference(lazy_extension, [assumptions([a21, a22])], [c26, c27])).

cnf(a23, assumption,
	X16 = X14).
cnf(c29, plain,
	sK1(X16) != X15,
	inference(reflexivity, [assumptions([a23])], [c28])).

cnf(c30, axiom,
	hates(agatha,X4) | X4 = butler
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_7)])).
cnf(a24, assumption,
	sK1(X16) = X17).
cnf(c31, plain,
	hates(agatha,X17),
	inference(variable_backward_paramodulation, [assumptions([a24])], [c29, c30])).
cnf(c32, plain,
	butler != X18 | X18 != X15,
	inference(variable_backward_paramodulation, [assumptions([a24])], [c29, c30])).

cnf(a25, assumption,
	butler = X18).
cnf(c33, plain,
	X18 != X15,
	inference(reflexivity, [assumptions([a25])], [c32])).

cnf(a26, assumption,
	X18 = X15).
cnf(c34, plain,
	$false,
	inference(reflexivity, [assumptions([a26])], [c33])).

cnf(c35, axiom,
	hates(butler,X4) | ~hates(agatha,X4)
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_9)])).
cnf(a27, assumption,
	agatha = agatha).
cnf(a28, assumption,
	X17 = X19).
cnf(c36, plain,
	hates(butler,X19),
	inference(strict_extension, [assumptions([a27, a28])], [c31, c35])).

cnf(c37, plain,
	~hates(X16,sK1(X16))).
cnf(a29, assumption,
	butler = X16).
cnf(a30, assumption,
	X19 = sK1(X16)).
cnf(c38, plain,
	$false,
	inference(reduction, [assumptions([a29, a30])], [c36, c37])).

cnf(c39, axiom,
	lives(sK0)
	inference(clausify, [status(esa)], [file('Problems/PUZ/PUZ001+1.p', pel55_1)])).
cnf(a31, assumption,
	X1 = sK0).
cnf(c40, plain,
	$false,
	inference(strict_extension, [assumptions([a31])], [c21, c39])).

cnf(c41, plain,
	$false,
	inference(constraint_solving, [
		bind(X1, sK0),
		bind(X2, sK0),
		bind(X3, charles),
		bind(X5, charles),
		bind(X6, agatha),
		bind(X7, agatha),
		bind(X8, agatha),
		bind(X9, butler),
		bind(X10, butler),
		bind(X11, butler),
		bind(X12, agatha),
		bind(X13, butler),
		bind(X16, butler),
		bind(X14, butler),
		bind(X15, butler),
		bind(X17, sK1(X16)),
		bind(X18, butler),
		bind(X19, sK1(X16))
	],
	[a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31])).

% SZS output end CNFRefutation for PUZ001+1
```

[1] Paskevich, Andrei. "Connection tableaux with lazy paramodulation." Journal of Automated Reasoning 40.2-3 (2008): 179-194.
