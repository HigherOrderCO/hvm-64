# HVM-Core: a parallel Interaction Combinator evaluator

HVM-Core is a parallel evaluator for extended [Symmetric Interaction Combinators](https://www-lipn.univ-paris13.fr/~mazza/papers/CombSem-MSCS.pdf).

We provide a raw syntax for specifying nets, a reference implementation in Rust,
and a massively parallel evaluator in CUDA. In our benchmarks, HVMC performs up
to **6.8 billion interactions** per second in an RTX 4090. When evaluating
functional programs, it performed 5x faster than the best alternatives, making
it a great compile target for high-level languages seeking massive parallelism.

HVM-Core will be used as the compile target for [HVM](https://github.com/higherorderco/hvm)
on its upcoming update. Guidelines for language implementers will be published soon.

## Usage

Install HVM-Core as:

```
cargo install hvm-core
```

Then, run the reference single-core implementation as:

```
hvmc run file.hvmc -s
```

If you have a GPU, run it with thousands of threads as:

```
hvmc bend file.hvmc -s
```

**Note**: `bend` wasn't implemented yet; if you're here *today* and wants to run
it on the GPU, you'll need to manually edit `hvm2.cu` using the
`hvmc gen-cuda-book file.hvmc` command. Compile with `-arch=compute_89`. I'll
add that to the CLI soon. Accepting PRs though :)

## Example

HVMC is a low-level compile target for high-level languages. The
file below performs 2^2 with [Church Nats](https://en.wikipedia.org/wiki/Church_encoding):

```javascript
// closed net for the Church Nat 2
@c2_a = ([(b a) (a R)] (b R))

// also 2, but with a different label
@c2_b = ({2 (b a) (a R)} (b R))

// applies 2 to 2 to exponentiate
@main = ret
      & @c2_a ~ (@c2_b ret)
```

To understand this syntax, see below. More on [`/examples`](/examples).

## Language

HVM-Core provides a textual syntax that allows us to construct interaction
combinator nets using a simple AST:

```
<TERM> ::=
  <ERA> ::= "*"
  <CON> ::= "(" <TERM> " " <TERM> ")"
  <DUP> ::= "[" <TERM> " " <TERM> "]"
  <CTR> ::= "{" <label> " " <TERM> " " <TERM> "}"
  <REF> ::= "@" <name>
  <U24> ::= "#" <value>
  <OP2> ::= "<" <TERM> " " <TERM> ">"
  <ITE> ::= "?" <TERM> <TERM>
  <VAR> ::= <name>

<NET> ::=
  <ROOT> ::= <TERM>
  <RDEX> ::= "&" <TERM> "~" <TERM> <NET>

<BOOK> ::= 
  <DEF> ::= "@" <NET> <BOOK>
  <END> ::= <EOF> 
```

On top of pure interaction combinators, HVMC includes a minimal set of
performance-critical features, including top-level definitions (as closed nets),
unboxed 24-bit numbers, binary numeric operations and if-then-else. Below is a
complete list of all term variants:

- `ERA`: an eraser node, as defined on the reference system.

- `CON`: a constructor node, as defined on the reference system.

- `DUP`: a duplicator node, as defined on the reference system.

- `CTR`: an "extra" node, which behaves exactly like CON/DUP nodes, but with a
  different symbol. When the label is 0/1, it corresponds to a CON/DUP node.

- `VAR`: a named variable, used to create a wire. Each name must occur twice,
  denoting both endpoints of a wire.

- `REF`: a reference to a top-level definition, which is itself a closed net.
  That reference is unrolled lazily, allowing for recursive functions to be
  implemented without the need for Church numerals and the like.

- `U24`: an unboxed 24-bit unsigned integer.

- `OP2`: a binary operation on u24 operands.

- `ITE`: an if-then-else operator with a u24 condition.

Note that terms form a tree-like structure. Yet, interaction combinators are not
trees, but graphs; terms aren't enough to express all possible nets. To fix
that, we provide the `& <TERM> ~ <TERM>` syntax, which connects the top-most
main port of each tree. This allows us to build any closed nets with a single
free wire. For example, to build the closed net:

```
R       .............
:       :           :
:      /_\         /_\
:    ..: :..     ..: :..
:    :     :     :     :
:   /_\   /_\   /_\   /_\
:...: :   : :...: :   :.:
      *   :.......:
```

We could use the following syntax:

```
@main
  = R
  & ((R *) (x y))
  ~ ((y x) (z z))
```

Here, `@main` is the name of the closed net, `R` is used to denote its single
free wire, each CON node is denoted as `(x y)`, and the ERA node is represented
as a `*`. The wires from an aux port to a main port are denoted by the tree
hierarchy, the wires between aux ports are denoted by named variables, and the
single wire between main ports is denoted by the `& A ~ B` syntax. Note this
always represents an active pair (or redex)!

## Evaluator

HVMC comes with 2 evaluators: a reference interpreter in Rust, and a massively
parallel runtime in CUDA. Both evaluators are completely eager, which means they
will reduce *every* generated active pair (redex) in an ultra-greedy fashion.
Because of that, to perform recursion, it is advisable to pre-compile the source
language to a [supercombinator](https://en.wikipedia.org/wiki/Supercombinator)
formulation, as it allows sub-expressions to be unrolled lazily, preventing HVMC
from infinitely expanding recursive function bodies. For the same reason, terms
like the Y-Combinator aren't compatible with current HVMC. A lazy evaluator
version will be provided in the future.

The eager evaluator works by keeping a vector of current active pairs (redexes)
and, for-each redex, performing an "interaction", as described below. On the
single-core version, that "for-each" is done in a sequential loop. On the
multi-core version, the vector of redexes is actually split into a grid of
"redex bags", each bag owned by a "rewrite squad", which continuously pops and
executes a redex in parallel. Since interaction rules are symmetric on the 4
surrounding ports, we actually use 4 threads to perform an interaction, thus the
name "squad".

For example, on NVidia's RTX 4090, we keep a grid of 128x128 redex bags. Each
bag is processed by a rewrite squad. There are `16,384` active squads, for a
total of `65,536` active threads. Below is a visual representation:

```
REDEX ::= (Ptr32, Ptr32)

    A1 --|\        /|-- B2
         | |A0--B0| |
    A2 --|/        \|-- B1

REDEX_BAG ::= Vec<REDEX>

    [(A0,B0), (C0,D0), (E0,F0), ...]

REDEX_GRID ::= Matrix<128, 128 REDEX_BAG>

    [ ] -- [ ] -- [ ] -- [ ] -- [ ] -- [ ] -- [ ] -- [ ] ...
     |      |      |      |      |      |      |      |
    [ ] -- [ ] -- [ ] -- [ ] -- [ ] -- [ ] -- [ ] -- [ ] ...
     |      |      |      |      |      |      |      |
    [ ] -- [ ] -- [ ] -- [ ] -- [ ] -- [ ] -- [ ] -- [ ] ...
     |      |      |      |      |      |      |      |
    [ ] -- [ ] -- [ ] -- [ ] -- [ ] -- [ ] -- [ ] -- [ ] ...
     |      |      |      |      |      |      |      |
    [ ] -- [ ] -- [ ] -- [ ] -- [ ] -- [ ] -- [ ] -- [ ] ...
     |      |      |      |      |      |      |      |
    [ ] -- [ ] -- [ ] -- [ ] -- [ ] -- [ ] -- [ ] -- [ ] ...
     |      |      |      |      |      |      |      |
    [ ] -- [ ] -- [ ] -- [ ] -- [ ] -- [ ] -- [ ] -- [ ] ...
     |      |      |      |      |      |      |      |
    [ ] -- [ ] -- [ ] -- [ ] -- [ ] -- [ ] -- [ ] -- [ ] ...
     |      |      |      |      |      |      |      |
    ...    ...    ...    ...    ...    ...    ...    ... 

SQUAD ::= Vec<4, THREAD>

THREAD ::=

    loop {
      if (A, B) = pop_redex():
        atomic_rewrite(A, B)
      share_redexes()
    }

    atomic_rewrite(A, B):
      if con-con:
        thread 0: atomic_subst(A.1, B.1)
        thread 1: atomic_subst(A.2, B.2)
        thread 2: atomic_subst(B.1, A.1)
        thread 3: atomic_subst(B.2, A.2)
      else if con-dup:
        ...

    atomic_subst(a, b):
      a_ok = CAS(a, a.rev(), b)
      b_ok = CAS(b, b.rev(), a)
      if not (a_ok and b_ok):
        atomic_link(a, a.rev(), b)

    atomic_link(a, b):
      see algorith on 'paper/'

RESULT:

    - thread 0 links A1 -> B1
    - thread 1 links B1 -> A1
    - thread 2 links A2 -> B2
    - thread 3 links B2 -> A2

OUTPUT:

    A1 <--------------> B1
    A2 <--------------> B2
```

With this setup, a program reaches the peak parallelism when there are about 16k
active redexes, resulting in a maximum theoretical speedup of 65536x. Note that
even a "maximally sequential" interaction combinator program (i.e., one that, at
any point of the reduction, always has exactly 1 active pair) still benefits
from a "4x" degree of parallelism, due to squads having 4 threads; although,
obviously, at that point a CPU version would be much faster, as CPU cores are
way faster than 4 GPU cores.

## Interactions

The available interactions are the same described on the reference paper, namely
annihilation and commutation rules, plus a few additional interactions,
including primitive numeric operations, conditionals, and dereference. The core
interactions are:

```
()---()
~~~~~~~ ERA-ERA
nothing
```

```
A1 --|\
     |a|-- ()
A2 --|/
~~~~~~~~~~~~~ CTR-ERA
A1 ------- ()
A2 ------- ()
```

```
A1 --|\     /|-- B2
     |a|---|b|   
A2 --|/     \|-- B1
~~~~~~~~~~~~~~~~~~~ CTR-CTR (if a ~~ b)
A1 -----, ,----- B2
         X
A2 -----' '----- B1
```

```
A1 --|\         /|-- B2
     |a|-------|b|   
A2 --|/         \|-- B1
~~~~~~~~~~~~~~~~~~~~~~~ CTR-CTR (if a !~ b)
      /|-------|\
A1 --|b|       |a|-- B2
      \|--, ,--|/
           X
      /|--' '--|\
A2 --|b|       |a|-- B1
      \|-------|/
```

The dereference interactions happen when a @REF node interacts with another
node. When that node is a constructor, the dereference will be unrolled
efficiently. This makes HVM practical, because, without it, top-level
definitions would need to be implemented with DUP nodes. This would cause
considerable overhead when trying to implement functions, due to DUP nodes
incremental copying nature. When the other node is anything else, that implies
two closed nets got disconnected from the main graph, so both nodes are
collected, allowing recursive functions to halt without infinite expansions.

```
() -- @REF
~~~~~~~~~~ ERA-REF
nothing
```

```
A1 --|\
     | |-- @REF
A2 --|/
~~~~~~~~~~~~~~~~ CTR-REF
A1 --|\
     | |-- {val}
A2 --|/
```

Since Interaction Combinator nodes only have 1 active port, which is a property
that is essential for key computational characteristics such as strong
confluence, we can't have a binary numeric operation node. Instead, we split
numeric operations in two nodes: OP2, which processes the first operand and
returns an OP1 node, which then processes the second operand, performs the
computation, and connects the result to the return wire.

```
A1 --,
     [}-- #X
A2 --' 
~~~~~~~~~~~~~~ OP2-NUM
A2 --[#X}-- A1
```

```
A1 --[#X}-- #Y
~~~~~~~~~~~~~~ OP1-NUM
A1 -- #Z
```

Note that the OP2 operator doesn't store the operation type. Instead, it is
stored on 4 unused bits of the left operand. As such, an additional operation
called "load-op-type" is used to load the next operation on the left operand.
See the `/examples` directory for more info. Below is a table with all available
operations:

N   | operation
--- | ---------
  0 | load-op-type
  1 | addition
  2 | subtraction
  3 | multiplication
  4 | division
  5 | modulus
  6 | equal-to
  7 | not-equal-to
  8 | less-than
  9 | greater-than
  A | logical-and
  B | logical-or
  C | logical-xor
  D | logical-not
  E | left-shift
  F | right-shift

Since HVM already provides plenty of solutions for branching (global references,
lambda encoded booleans and pattern-matching, etc.), the If-Then-Else operation
is only necessary to read bits from numbers: otherwise, numbers would be "black
boxes" that can't interact with the rest of the program. The way it works is
simple: it receives a number, two branches (in a CON node) and a return wire. If
the number is 0, it erases the first branch and returns the second. Otherwise,
it does the opposite. Below is the reduction when #X is 1:

```
A1 --,
     (?)-- #X
A2 --' 
~~~~~~~~~~~~~ ITE-NUM
      /|-- ()
A1 --| |
      \|-- A2
```

Note that some interactions like NUM-ERA are omitted, but should logically
follow from the ones described above.

## Memory Layout

The memory layout is optimized for efficiency, and can be summarized as:

```rust
// A pointer is a 32-bit word
type Ptr = u32;

// A node stores its two aux ports
struct Node {
  p1: Ptr, // this node's fst aux port
  p2: Ptr, // this node's snd aux port 
}

// A redex links two main ports
struct Redex {
  a: Ptr, // main port of node A
  b: Ptr, // main port of node B
}

// A closed net
struct Net {
  root: Ptr,       // a free wire
  rdex: Vec<Redex> // a vector of redexes
  heap: Vec<Node>  // a vector of nodes
}
```

As you can see, the memory layout resembles the textual syntax, with nets being
represented as a vector of trees, with the 'redex' buffer storing the tree roots
(as active pairs), and the 'nodes' buffer storing all the nodes. Each node has
two 32-bit pointers and, thus, uses exactly 64 bits. Pointers include a 4-bit
tag and a 28-bit value, which allows addressing a 2 GB space per instance. There
are 16 pointer types:

```rust
VR1 = 0x0; // variable to aux port 1
VR2 = 0x1; // variable to aux port 2
RD1 = 0x2; // redirect to aux port 1
RD2 = 0x3; // redirect to aux port 2
REF = 0x4; // lazy closed net
ERA = 0x5; // unboxed eraser
NUM = 0x6; // unboxed number
OP2 = 0x7; // numeric operation (binary)
OP1 = 0x8; // numeric operation (unary)
ITE = 0x9; // numeric if-then-else
CT0 = 0xA; // main port of con
CT1 = 0xB; // main port of dup
CT2 = 0xC; // main port of extra node 
CT3 = 0xD; // main port of extra node
CT4 = 0xE; // main port of extra node
CT5 = 0xF; // main port of extra node
```

This memory-efficient format allows for a fast implementation in many
situations. For example, allocation can be performed by a single 64-bit atomic
CAS, and an annihilation interaction can be done with 2 calls to an
`atomic_link()` procedure, which, in most cases, only requires 2 atomic reads
and 2 atomic CAS's.

We also provide unboxed 24-bit unsigned integers, which allows HVMC to store raw
data with minimal loss. For example, to store a raw 3 KB buffer, one could use a
perfect binary tree of CON nodes with a depth of 9, as follows:

```javascript
@buff = ((((((((((X0 X1) (X2 X3)) ((X4 X5) (X6 X7))) ...)))))))
```

This would use a total of 1023 nodes, which takes a space of almost exactly 8 KB
on HVMC. As such, while buffers are not part of the spec, we can store raw data
with a 37.5% efficiency using pure interaction nets, which is a reasonable price
to pay to gain the ability of computing them with massive parallelism.
