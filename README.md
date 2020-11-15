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
lazyCoP understands the [TPTP](http://tptp.org) FOF and CNF dialects.
`lazycop --help` will print a help message.
After a problem has been loaded successfully and preprocessed, lazyCoP will attempt to prove it.

One of three things will then happen:
 1. lazyCoP finds a proof, which it will print to standard output. Hooray!
 2. lazyCoP exhausts its search space without finding a proof. This means that the conjecture does not follow from the axioms, or that the set of clauses is satisfiable.
 3. lazyCoP runs until you get bored. If a proof exists, it will be found eventually, but perhaps not in reasonable time.

Proofs are expressed as an unsatisfiable set of ground clauses.
A typical run on [PUZ001+1](http://tptp.org/cgi-bin/SeeTPTP?Category=Problems&Domain=PUZ&File=PUZ001+1.p):
```
$ lazycop Problems/PUZ/PUZ001+1.p
% SZS status Theorem for PUZ001+1
% SZS output begin Proof for PUZ001+1
cnf(0, negated_conjecture,
	~killed(agatha,agatha),
	file('Problems/PUZ/PUZ001+1.p', pel55)).
cnf(1, axiom,
	sK0 = agatha | sK0 = charles | sK0 = butler | ~lives(sK0),
	file('Problems/PUZ/PUZ001+1.p', pel55_3)).
cnf(2, axiom,
	killed(sK0,agatha),
	file('Problems/PUZ/PUZ001+1.p', pel55_1)).
cnf(3, axiom,
	killed(sK0,agatha),
	file('Problems/PUZ/PUZ001+1.p', pel55_1)).
cnf(4, axiom,
	~killed(charles,agatha) | hates(charles,agatha),
	file('Problems/PUZ/PUZ001+1.p', pel55_4)).
cnf(5, axiom,
	~hates(charles,agatha) | ~hates(agatha,agatha),
	file('Problems/PUZ/PUZ001+1.p', pel55_6)).
cnf(6, axiom,
	hates(agatha,agatha) | agatha = butler,
	file('Problems/PUZ/PUZ001+1.p', pel55_7)).
cnf(7, axiom,
	agatha != butler,
	file('Problems/PUZ/PUZ001+1.p', pel55_11)).
cnf(8, axiom,
	killed(sK0,agatha),
	file('Problems/PUZ/PUZ001+1.p', pel55_1)).
cnf(9, axiom,
	~killed(butler,agatha) | ~richer(butler,agatha),
	file('Problems/PUZ/PUZ001+1.p', pel55_5)).
cnf(10, axiom,
	richer(butler,agatha) | hates(butler,butler),
	file('Problems/PUZ/PUZ001+1.p', pel55_8)).
cnf(11, axiom,
	~hates(butler,sK1(butler)),
	file('Problems/PUZ/PUZ001+1.p', pel55_10)).
cnf(12, axiom,
	sK1(butler) = butler | hates(agatha,sK1(butler)),
	file('Problems/PUZ/PUZ001+1.p', pel55_7)).
cnf(13, axiom,
	~hates(agatha,sK1(butler)) | hates(butler,sK1(butler)),
	file('Problems/PUZ/PUZ001+1.p', pel55_9)).
cnf(14, axiom,
	lives(sK0),
	file('Problems/PUZ/PUZ001+1.p', pel55_1)).
% SZS output end Proof for PUZ001+1
% problem symbols	: 9
% problem clauses	: 15
% eliminated leaves	: 4297
% retained leaves	: 2615
% expanded leaves	: 1009
$
```

## References
1. Paskevich, Andrei. "Connection tableaux with lazy paramodulation." Journal of Automated Reasoning 40.2-3 (2008): 179-194.
2. Otten, Jens. "Restricting backtracking in connection calculi." AI Communications 23.2-3 (2010): 159-182.
