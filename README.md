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
 - tautology elimination: no clause added to the tableau may contain a tautology
 - path regularity: no literal can occur twice in any path
 - enforced _hochklappen_ ("folding up"): refutations of unit literal "lemmas" are available for re-use where possible
 - strong regularity: a branch cannot be extended if its leaf literal could be closed by reduction with a path literal or lemma

See the _Handbook of Automated Reasoning (Vol. II)_, "Model Elimination and Connection Tableau Procedures" for more information and an introduction to the predicate calculus.

As well as these predicate refinements, lazyCoP implements some obvious modifications for equality literals:
 - reflexivity elimination: no clause may contain `s = t` if `s`, `t` are identical
 - reflexive regularity: if `s != t` is a literal in the tableau and `s`, `t` are identical, it must be closed immediately
 - symmetric path regularity: neither `s = t` nor `t = s` may appear in a path if `s = t` does
 - strong equality regularity: a branch cannot be extended by an equality if said equality is available by path or lemma
 - superposition-style ordering constraints: if `s[p] = u` is the target of a paramodulation, `s[p] > u`

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

A prebuilt static binary for Linux-like platforms is available.
A processor supporting AVX instructions is required (but this is not necessary if building from source).

## Usage

lazyCoP reads the [TPTP](http://tptp.org) CNF dialect from the standard input.
To load problems which are not in clause normal form, use an external clausifier system such as [Vampire](https://vprover.github.io) or [E](https://eprover.org).
After a problem has been loaded successfully and standard input has closed, lazyCoP will attempt to prove it.
To do this it will employ all available CPU cores and a steadily-growing amount of memory.

One of three things will then happen:
 1. lazyCoP finds a proof, which it will print to standard output in [TSTP](http://www.tptp.org/TSTP/) format. Hooray!
 2. lazyCoP exhausts its search space without finding a proof. This means that the conjecture does not follow from the axioms, or that the set of clauses is satisfiable. This is reported `SZS status Unknown` for technical reasons.
 3. lazyCoP runs until you get bored. If a proof exists, it will be found eventually, but perhaps not in reasonable time.
 
A typical run on [PUZ001+1](http://tptp.org/cgi-bin/SeeTPTP?Category=Problems&Domain=PUZ&File=PUZ001+1.p):
```
$ ./clausify PUZ001+1.p | lazycop
<some thinking occurs...>
% SZS status Unsatisfiable
% SZS output begin IncompleteProof
cnf(c0, axiom,
	~killed(agatha,agatha)).
cnf(c1, plain,
	~killed(agatha,agatha),
	inference(start, [], [c0])).

cnf(c2, axiom,
	charles = X0 | butler = X0 | agatha = X0 | ~lives(X0)).
cnf(a0, assumption,
	agatha = agatha).
cnf(c3, plain,
	$false,
	inference(strict_function_extension, [assumptions([a0])], [c1, c2])).
cnf(c4, plain,
	charles = X0 | butler = X0 | ~lives(X0),
	inference(strict_function_extension, [assumptions([a0])], [c1, c2])).
cnf(c5, plain,
	X1 != X0 | ~killed(X1,agatha),
	inference(strict_function_extension, [assumptions([a0])], [c1, c2])).

cnf(a1, assumption,
	X1 = X0).
cnf(c6, plain,
	~killed(X1,agatha),
	inference(reflexivity, [assumptions([a1])], [c5])).

cnf(c7, axiom,
	killed(sK0,agatha)).
cnf(a2, assumption,
	X1 = sK0).
cnf(a3, assumption,
	agatha = agatha).
cnf(c8, plain,
	$false,
	inference(strict_predicate_extension, [assumptions([a2, a3])], [c6, c7])).
cnf(c9, plain,
	$false,
	inference(strict_predicate_extension, [assumptions([a2, a3])], [c6, c7])).

cnf(c10, axiom,
	~hates(charles,X2) | ~hates(agatha,X2)).
cnf(a4, assumption,
	charles = charles).
cnf(c11, plain,
	butler = X0 | ~lives(X0),
	inference(strict_subterm_extension, [assumptions([a4])], [c4, c10])).
cnf(c12, plain,
	~hates(agatha,X2),
	inference(strict_subterm_extension, [assumptions([a4])], [c4, c10])).
cnf(c13, plain,
	~hates(X0,X2),
	inference(strict_subterm_extension, [assumptions([a4])], [c4, c10])).

cnf(c14, axiom,
	hates(X3,X4) | ~killed(X3,X4)).
cnf(a5, assumption,
	X0 = X3).
cnf(a6, assumption,
	X2 = X4).
cnf(c15, plain,
	$false,
	inference(strict_predicate_extension, [assumptions([a5, a6])], [c13, c14])).
cnf(c16, plain,
	~killed(X3,X4),
	inference(strict_predicate_extension, [assumptions([a5, a6])], [c13, c14])).

cnf(c17, plain,
	killed(X1,agatha)).
cnf(a7, assumption,
	X3 = X1).
cnf(a8, assumption,
	X4 = agatha).
cnf(c18, plain,
	$false,
	inference(predicate_reduction, [assumptions([a7, a8])], [c16, c17])).

cnf(c19, axiom,
	hates(agatha,X5) | butler = X5).
cnf(a9, assumption,
	agatha = agatha).
cnf(a10, assumption,
	X2 = X5).
cnf(c20, plain,
	$false,
	inference(strict_predicate_extension, [assumptions([a9, a10])], [c12, c19])).
cnf(c21, plain,
	butler = X5,
	inference(strict_predicate_extension, [assumptions([a9, a10])], [c12, c19])).

cnf(c22, axiom,
	agatha != butler).
cnf(a11, assumption,
	butler = butler).
cnf(c23, plain,
	$false,
	inference(strict_subterm_extension, [assumptions([a11])], [c21, c22])).
cnf(c24, plain,
	$false,
	inference(strict_subterm_extension, [assumptions([a11])], [c21, c22])).
cnf(c25, plain,
	agatha != X5,
	inference(strict_subterm_extension, [assumptions([a11])], [c21, c22])).

cnf(a12, assumption,
	agatha = X5).
cnf(c26, plain,
	$false,
	inference(reflexivity, [assumptions([a12])], [c25])).

cnf(c27, axiom,
	hates(agatha,X6) | butler = X6).
cnf(a13, assumption,
	butler = butler).
cnf(c28, plain,
	~lives(X0),
	inference(strict_subterm_extension, [assumptions([a13])], [c11, c27])).
cnf(c29, plain,
	hates(agatha,X6),
	inference(strict_subterm_extension, [assumptions([a13])], [c11, c27])).
cnf(c30, plain,
	X0 = X6,
	inference(strict_subterm_extension, [assumptions([a13])], [c11, c27])).

cnf(c31, axiom,
	~hates(X7,sK1(X7))).
cnf(a14, assumption,
	sK1(X7) = X6).
cnf(c32, plain,
	$false,
	inference(strict_subterm_extension, [assumptions([a14])], [c30, c31])).
cnf(c33, plain,
	$false,
	inference(strict_subterm_extension, [assumptions([a14])], [c30, c31])).
cnf(c34, plain,
	~hates(X7,X0),
	inference(strict_subterm_extension, [assumptions([a14])], [c30, c31])).

cnf(c35, axiom,
	hates(butler,X8) | richer(X8,agatha)).
cnf(a15, assumption,
	X7 = butler).
cnf(a16, assumption,
	X0 = X8).
cnf(c36, plain,
	$false,
	inference(strict_predicate_extension, [assumptions([a15, a16])], [c34, c35])).
cnf(c37, plain,
	richer(X8,agatha),
	inference(strict_predicate_extension, [assumptions([a15, a16])], [c34, c35])).

cnf(c38, axiom,
	~richer(X9,X10) | ~killed(X9,X10)).
cnf(a17, assumption,
	X8 = X9).
cnf(a18, assumption,
	agatha = X10).
cnf(c39, plain,
	$false,
	inference(strict_predicate_extension, [assumptions([a17, a18])], [c37, c38])).
cnf(c40, plain,
	~killed(X9,X10),
	inference(strict_predicate_extension, [assumptions([a17, a18])], [c37, c38])).

cnf(c41, plain,
	killed(X1,agatha)).
cnf(a19, assumption,
	X9 = X1).
cnf(a20, assumption,
	X10 = agatha).
cnf(c42, plain,
	$false,
	inference(predicate_reduction, [assumptions([a19, a20])], [c40, c41])).

cnf(c43, axiom,
	hates(butler,X11) | ~hates(agatha,X11)).
cnf(a21, assumption,
	agatha = agatha).
cnf(a22, assumption,
	X6 = X11).
cnf(c44, plain,
	$false,
	inference(strict_predicate_extension, [assumptions([a21, a22])], [c29, c43])).
cnf(c45, plain,
	hates(butler,X11),
	inference(strict_predicate_extension, [assumptions([a21, a22])], [c29, c43])).

cnf(c46, axiom,
	~hates(X12,sK1(X12))).
cnf(a23, assumption,
	butler = X12).
cnf(a24, assumption,
	X11 = sK1(X12)).
cnf(c47, plain,
	$false,
	inference(strict_predicate_extension, [assumptions([a23, a24])], [c45, c46])).
cnf(c48, plain,
	$false,
	inference(strict_predicate_extension, [assumptions([a23, a24])], [c45, c46])).

cnf(c49, axiom,
	lives(sK0)).
cnf(a25, assumption,
	X0 = sK0).
cnf(c50, plain,
	$false,
	inference(strict_predicate_extension, [assumptions([a25])], [c28, c49])).
cnf(c51, plain,
	$false,
	inference(strict_predicate_extension, [assumptions([a25])], [c28, c49])).

cnf(c52, plain,
	$false,
	inference(constraint_solving, [
		bind(X0, sK0),
		bind(X1, sK0),
		bind(X2, agatha),
		bind(X3, sK0),
		bind(X4, agatha),
		bind(X5, agatha),
		bind(X6, sK1(X7)),
		bind(X7, butler),
		bind(X8, sK0),
		bind(X9, sK0),
		bind(X10, agatha),
		bind(X11, sK1(X7)),
		bind(X12, butler)
	],
	[a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25])).

% SZS output end IncompleteProof
```

[1] Paskevich, Andrei. "Connection tableaux with lazy paramodulation." Journal of Automated Reasoning 40.2-3 (2008): 179-194.
